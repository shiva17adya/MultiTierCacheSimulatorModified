"""Physical tier policy for HA-QM-ARC.

Implements history-aware tier-to-tier demotion and eviction guarantees.
"""

import math
from simpy.core import Environment

from common.packet import Packet
from forwarder_structures.content_store.tier import Tier
from forwarder_structures.forwarder import Forwarder
from policies.QM_ARC.tier_qm_arc_policy import QMARCPolicy


class TierHAQMARCPolicy(QMARCPolicy):
    """Physical storage tier policy for HA-QM-ARC."""

    def __init__(self, env: Environment, forwarder: Forwarder, tier: Tier):
        super().__init__(env, forwarder, tier)

    def get_effective_p(self) -> float:
        """Calculate local effective p scaled to this tier's capacity."""
        if self.forwarder.tiers and self.forwarder.tiers[0].strategies:
            coordinator = self.forwarder.tiers[0].strategies[0]
            if (
                hasattr(coordinator, "adaptive_controller")
                and hasattr(coordinator, "history_manager")
                and hasattr(coordinator, "c")
                and coordinator.c > 0
            ):
                delta_p_global = coordinator.adaptive_controller.compute_delta_p_history(
                    coordinator.c,
                    coordinator.history_manager.get_burst_score(),
                    coordinator.history_manager.get_scan_score(),
                    coordinator.history_manager.get_ghr_gradient(),
                )
                tier_ratio = float(self.c) / float(coordinator.c)
                delta_p_tier = round(delta_p_global * tier_ratio)
                return max(0.0, min(float(self.c), self.p + delta_p_tier))
        return float(self.p)

    def on_packet_access_t1(self, env: Environment, res, packet: Packet, index: int = -1):
        """Write packet to t1, demoting to lower tier if capacity is reached."""
        print('%s arriving at %s' % (self.tier.name, env.now))

        eff_p = self.get_effective_p()
        if len(self.t1) + len(self.t2) >= self.c:
            if self.t1 and len(self.t1) >= eff_p:
                yield env.process(self.send_to_next_level_t1(env, res))
            elif self.t2:
                yield env.process(self.send_to_next_level_t2(env, res))
            else:
                yield env.process(self.send_to_next_level_t1(env, res))

        if index == -1:
            print('insert into %s at MRU pos' % self.tier.name)
            self.t1.append_left(packet.name, packet)
            yield env.process(self.forwarder.index.update_packet_tier(packet.name, self.tier))
        else:
            print('insert into %s at pos :%s' % (self.tier.name, index))
            self.t1.append_by_index(index, packet.name, packet)
            yield env.process(self.forwarder.index.update_packet_tier(packet.name, self.tier))

        self.tier.number_of_packets += 1
        self.tier.number_of_write += 1
        self.tier.used_size += packet.size

        with res[self.forwarder.tiers.index(self.tier)].request() as req:
            yield req
            print('%s starting at %s' % (self.tier.name, env.now))
            yield env.timeout(self.tier.latency + packet.size / self.tier.write_throughput)
            self.tier.time_spent_writing += self.tier.latency + packet.size / self.tier.write_throughput
            res[self.forwarder.tiers.index(self.tier)].release(req)

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s leaving the resource at %s' % (self.tier.name, env.now))
            return

    def on_packet_access_t2(self, env: Environment, res, packet: Packet, index: int = -1):
        """Write packet to t2, demoting to lower tier if capacity is reached."""
        print('%s arriving at %s' % (self.tier.name, env.now))

        eff_p = self.get_effective_p()
        if len(self.t1) + len(self.t2) >= self.c:
            if self.t1 and len(self.t1) >= eff_p:
                yield env.process(self.send_to_next_level_t1(env, res))
            elif self.t2:
                yield env.process(self.send_to_next_level_t2(env, res))
            else:
                yield env.process(self.send_to_next_level_t1(env, res))

        if index == -1:
            self.t2.append_left(packet.name, packet)
            yield env.process(self.forwarder.index.update_packet_tier(packet.name, self.tier))
        else:
            print("write to index : %s" % index)
            self.t2.append_by_index(index, packet.name, packet)
            yield env.process(self.forwarder.index.update_packet_tier(packet.name, self.tier))

        self.tier.number_of_packets += 1
        self.tier.number_of_write += 1
        self.tier.used_size += packet.size

        with res[self.forwarder.tiers.index(self.tier)].request() as req:
            yield req
            print('%s starting at %s' % (self.tier.name, env.now))
            yield env.timeout(self.tier.latency + packet.size / self.tier.write_throughput)
            self.tier.time_spent_writing += self.tier.latency + packet.size / self.tier.write_throughput
            res[self.forwarder.tiers.index(self.tier)].release(req)

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s leaving the resource at %s' % (self.tier.name, env.now))
            return
