import math

from simpy.core import Environment

from common.deque import Deque
from common.packet import Packet
from forwarder_structures.content_store.tier import Tier
from forwarder_structures.forwarder import Forwarder
from policies.policy import Policy


class AbstractMARCPolicy(Policy):
    def __init__(self, env: Environment, forwarder: Forwarder, tier: Tier):
        Policy.__init__(self, env, forwarder, tier)

        self.c = self.c()  # cache size

        self.p = 0  # Target size for the list T1
        self.t1 = self.t1()  # T1: recent cache entries
        self.t2 = self.t2()  # T2: frequent entries

    def get_physical_tiers(self):
        """Return all registered physical storage tiers, filtering out the virtual BOSS coordinator tier."""
        physical = [t for t in self.forwarder.tiers if t.name != "BOSS" and t.max_size > 0]
        if not physical:
            physical = [t for t in self.forwarder.tiers if t.max_size > 0]
        return physical

    def get_tier_betas(self):
        """Calculate beta_i = size_i / size_1 for each physical tier i (tier 1 is the fastest/top tier)."""
        physical_tiers = self.get_physical_tiers()
        if not physical_tiers:
            return {}, 1.0

        top_tier = physical_tiers[0]
        top_size = float(top_tier.max_size) if top_tier.max_size > 0 else 1.0

        betas = {}
        for tier in physical_tiers:
            betas[tier] = float(tier.max_size) / top_size

        beta_sum = sum(betas.values())
        return betas, beta_sum

    @property
    def beta_ssd(self):
        pt = self.get_physical_tiers()
        return (pt[1].max_size / pt[0].max_size) if len(pt) > 1 and pt[0].max_size > 0 else 1.0

    @property
    def beta_disk(self):
        pt = self.get_physical_tiers()
        return (pt[2].max_size / pt[0].max_size) if len(pt) > 2 and pt[0].max_size > 0 else self.beta_ssd

    def _replace(self, env: Environment, packet: Packet):
        in_b2 = yield env.process(self.forwarder.index.packet_in_ghost(packet.name, 'b2'))
        if self.t1 and ((in_b2 and len(self.t1) == self.p) or (len(self.t1) > self.p)):
            old_name, old = self.t1.get_without_pop()
            self.t1_pop(old)
            print("%s move from t1 to b1 " % old.name)
            yield env.process(self.forwarder.index.del_packet_from_cs(old.name))
            yield env.process(self.forwarder.index.update_packet_ghost(old.name, 'b1'))
        else:
            old_name, old = self.t2.get_without_pop()
            self.t2_pop(old)
            print("%s move from t2 to b2 " % old.name)
            yield env.process(self.forwarder.index.del_packet_from_cs(old.name))
            yield env.process(self.forwarder.index.update_packet_ghost(old.name, 'b2'))

    def on_packet_access(self, env: Environment, res, packet: Packet, is_write: bool):
        print('%s in %s' % (self.tier.name, env.now))
        if self.c == 0:
            print("cache units == %s" % self.c)
            return

        if not is_write and self.t1.__contains__(packet.name):
            print("%s hit in t1, promote to MRU t2" % packet.name)
            self.t1_remove(packet)
            yield env.process(self.t2_append_left(env, res, packet))

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s out of %s' % (self.tier.name, env.now))
            return

        if not is_write and self.t2.__contains__(packet.name):
            print("%s hit in t2, promote to MRU t2" % packet.name)
            self.t2_remove(packet)
            yield env.process(self.t2_append_left(env, res, packet))

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s out of %s' % (self.tier.name, env.now))
            return

        in_b1 = yield env.process(self.forwarder.index.packet_in_ghost(packet.name, 'b1'))
        if in_b1:
            len_b1 = yield env.process(self.forwarder.index.ghost_len('b1'))
            len_b2 = yield env.process(self.forwarder.index.ghost_len('b2'))
            self.increment_p(len_b1, len_b2)
            yield env.process(self._replace(env, packet))
            yield env.process(self.forwarder.index.del_packet_from_ghost(packet.name))
            print("%s hit in b1, promote to MRU t2" % packet.name)
            yield env.process(self.t2_append_left(env, res, packet))

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s out of %s' % (self.tier.name, env.now))
            return

        in_b2 = yield env.process(self.forwarder.index.packet_in_ghost(packet.name, 'b2'))
        if in_b2:
            len_b1 = yield env.process(self.forwarder.index.ghost_len('b1'))
            len_b2 = yield env.process(self.forwarder.index.ghost_len('b2'))
            self.decrement_p(len_b1, len_b2)
            yield env.process(self._replace(env, packet))
            yield env.process(self.forwarder.index.del_packet_from_ghost(packet.name))
            print("%s hit in b2, promote to MRU t2" % packet.name)
            yield env.process(self.t2_append_left(env, res, packet))

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s out of %s' % (self.tier.name, env.now))
            return

        len_b1 = yield env.process(self.forwarder.index.ghost_len('b1'))
        if len(self.t1) + len_b1 == self.c:
            print("%s cache miss in all queues" % packet.name)
            # Case A: L1 (T1 u B1) has exactly c pages.
            if len(self.t1) < self.c:
                print("evict from b1")
                yield env.process(self.forwarder.index.pop_packet_from_ghost('b1'))
                yield env.process(self._replace(env, packet))
            else:
                name, old = self.t1.get_without_pop()
                self.t1_pop(old)
                print("%s evict from t1" % old.name)
                yield env.process(self.forwarder.index.del_packet_from_cs(old.name))
        else:
            len_b1 = yield env.process(self.forwarder.index.ghost_len('b1'))
            len_b2 = yield env.process(self.forwarder.index.ghost_len('b2'))
            total = len(self.t1) + len_b1 + len(self.t2) + len_b2
            if total >= self.c:
                if total == (2 * self.c):
                    print("evict from b2")
                    yield env.process(self.forwarder.index.pop_packet_from_ghost('b2'))
                yield env.process(self._replace(env, packet))

        print("%s write to MRU t1" % packet.name)
        yield env.process(self.t1_append_left(env, res, packet))

        self.t1.__str__()
        self.t2.__str__()
        self.forwarder.index.__str__()
        self.forwarder.index.__str__(what="Ghost")
        print('%s out of %s' % (self.tier.name, env.now))
        return

    def t1(self):
        t1 = Deque()
        for tier in self.forwarder.tiers[1:]:
            for strategy in tier.strategies:
                try:
                    t1.update(strategy.t1)
                except Exception as e:
                    print(e)
        return t1

    def t2(self):
        t2 = Deque()
        for tier in self.forwarder.tiers[1:]:
            for strategy in tier.strategies:
                try:
                    t2.update(strategy.t2)
                except Exception as e:
                    print(e)
        return t2

    def c(self):
        c = 0
        for tier in self.forwarder.tiers[1:]:
            c += math.trunc(tier.max_size * tier.target_occupation / self.forwarder.slot_size)
        return c

    def t1_pop(self, old):
        self.t1.pop()
        for tier in reversed(self.forwarder.tiers[1:]):
            for strategy in tier.strategies:
                try:
                    if strategy.t1:
                        strategy.t1.remove(old.name)
                        # evict data
                        tier.number_of_eviction_from_this_tier += 1
                        tier.number_of_packets -= 1
                        tier.used_size -= old.size
                        break
                except Exception as e:
                    print(e)

    def t2_pop(self, old):
        self.t2.pop()
        for tier in reversed(self.forwarder.tiers[1:]):
            for strategy in tier.strategies:
                try:
                    if strategy.t2:
                        strategy.t2.remove(old.name)
                        # evict data
                        tier.number_of_eviction_from_this_tier += 1
                        tier.number_of_packets -= 1
                        tier.used_size -= old.size
                        break
                except Exception as e:
                    print(e)

    def t1_remove(self, packet):
        self.t1.remove(packet.name)
        for tier in self.forwarder.tiers[1:]:
            for strategy in tier.strategies:
                try:
                    if packet.name in strategy.t1:
                        strategy.t1.remove(packet.name)
                        tier.number_of_eviction_from_this_tier += 1
                        tier.number_of_packets -= 1
                        tier.used_size -= packet.size
                        break
                except Exception as e:
                    print(e)

    def t2_remove(self, packet):
        self.t2.remove(packet.name)
        for tier in self.forwarder.tiers[1:]:
            for strategy in tier.strategies:
                try:
                    if packet.name in strategy.t2:
                        strategy.t2.remove(packet.name)
                        tier.number_of_eviction_from_this_tier += 1
                        tier.number_of_packets -= 1
                        tier.used_size -= packet.size
                        break
                except Exception as e:
                    print(e)

    def t1_append_left(self, env, res, packet):
        self.t1.append_left(packet.name, packet)
        yield env.process(self.forwarder.get_default_tier().write_packet_t1(env, res, packet))

    def t2_append_left(self, env, res, packet):
        self.t2.append_left(packet.name, packet)
        yield env.process(self.forwarder.get_default_tier().write_packet_t2(env, res, packet))

    def increment_p(self, len_b1, len_b2):
        """Global and local size adaptation of T1 on hit in B1 (Algorithm 1, Function 3)."""
        betas, beta_sum = self.get_tier_betas()
        ratio = max(len_b2 / len_b1, 1.0) if len_b1 > 0 else 1.0
        delta_p = ratio * beta_sum
        self.p = min(self.c, self.p + delta_p)
        for tier, beta_i in betas.items():
            delta_pi = ratio * beta_i
            for strategy in tier.strategies:
                try:
                    strategy.p = min(strategy.c, strategy.p + delta_pi)
                except Exception as e:
                    print(e)

    def decrement_p(self, len_b1, len_b2):
        """Global and local size adaptation of T1 on hit in B2 (Algorithm 1, Function 3)."""
        betas, beta_sum = self.get_tier_betas()
        ratio = max(len_b1 / len_b2, 1.0) if len_b2 > 0 else 1.0
        delta_p = ratio * beta_sum
        self.p = max(0.0, self.p - delta_p)
        for tier, beta_i in betas.items():
            delta_pi = ratio * beta_i
            for strategy in tier.strategies:
                try:
                    strategy.p = max(0.0, strategy.p - delta_pi)
                except Exception as e:
                    print(e)
