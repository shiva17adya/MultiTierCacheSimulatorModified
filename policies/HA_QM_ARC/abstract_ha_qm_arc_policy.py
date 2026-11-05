"""Coordinator policy for HA-QM-ARC (History-Aware Adaptive QM-ARC).

Subclasses AbstractQMARCPolicy and integrates:
1. WorkloadHistoryManager for bounded O(1) streaming metrics tracking and outcome recording.
2. AdaptiveController for proactive p adaptation and dynamic QoS lambda placement.
"""

import math
from simpy.core import Environment

from common.packet import Packet
from common.penalty import get_alpha
from forwarder_structures.content_store.tier import Tier
from forwarder_structures.forwarder import Forwarder
from policies.QM_ARC.abstract_qm_arc_policy import AbstractQMARCPolicy
from policies.HA_QM_ARC.history_manager import WorkloadHistoryManager
from policies.HA_QM_ARC.adaptive_controller import AdaptiveController


class AbstractHAQMARCPolicy(AbstractQMARCPolicy):
    """Virtual BOSS coordinator policy for HA-QM-ARC."""

    def __init__(
        self,
        env: Environment,
        forwarder: Forwarder,
        tier: Tier,
        window_size: int = 100,
        burst_window_size: int = 20,
        scan_streak_threshold: int = 5,
        enable_scan_detection: bool = False,
        omega_burst: float = 0.15,
        omega_scan: float = 0.20,
        omega_grad: float = 0.10,
        kappa_qos: float = 0.50,
    ):
        super().__init__(env, forwarder, tier)

        self.history_manager = WorkloadHistoryManager(
            window_size=window_size,
            burst_window_size=burst_window_size,
            scan_streak_threshold=scan_streak_threshold,
            enable_scan_detection=enable_scan_detection,
        )

        self.adaptive_controller = AdaptiveController(
            omega_burst=omega_burst,
            omega_scan=omega_scan,
            omega_grad=omega_grad,
            kappa_qos=kappa_qos,
        )

    def get_effective_p(self) -> float:
        """Calculate effective target p incorporating history momentum bias."""
        return self.adaptive_controller.get_effective_p(
            base_p=self.p,
            c=self.c,
            burst_score=self.history_manager.get_burst_score(),
            scan_score=self.history_manager.get_scan_score(),
            ghr_gradient=self.history_manager.get_ghr_gradient(),
        )

    def _replace(self, env: Environment, packet: Packet):
        """History-aware replacement: evicts from T1 or T2 based on effective p."""
        in_b2 = yield env.process(self.forwarder.index.packet_in_ghost(packet.name, 'b2'))
        eff_p = self.get_effective_p()

        if self.t1 and ((in_b2 and len(self.t1) == eff_p) or (len(self.t1) > eff_p)):
            old_name, old = self.t1.get_without_pop()
            self.t1_pop(old)
            print("%s move from t1 to b1 [HA-QM-ARC eff_p=%.1f]" % (old.name, eff_p))
            yield env.process(self.forwarder.index.del_packet_from_cs(old.name))
            yield env.process(self.forwarder.index.update_packet_ghost(old.name, 'b1'))
        elif self.t2:
            old_name, old = self.t2.get_without_pop()
            self.t2_pop(old)
            print("%s move from t2 to b2 [HA-QM-ARC eff_p=%.1f]" % (old.name, eff_p))
            yield env.process(self.forwarder.index.del_packet_from_cs(old.name))
            yield env.process(self.forwarder.index.update_packet_ghost(old.name, 'b2'))
        elif self.t1:
            old_name, old = self.t1.get_without_pop()
            self.t1_pop(old)
            print("%s fallback move from t1 to b1" % old.name)
            yield env.process(self.forwarder.index.del_packet_from_cs(old.name))
            yield env.process(self.forwarder.index.update_packet_ghost(old.name, 'b1'))

    def on_packet_access(self, env: Environment, res, packet: Packet, is_write: bool):
        """Handle request access with streaming history recording and dynamic QoS."""
        print('%s in %s' % (self.tier.name, env.now))
        if self.c == 0:
            print("cache units == %s" % self.c)
            return

        # 1. Hit in T1
        if not is_write and self.t1.__contains__(packet.name):
            self.history_manager.record_access(
                packet.name, packet.priority, is_hit=True, hit_location="T1"
            )
            if packet.priority == "h":
                print("%s hit in t1, high priority, promote to MRU t2" % packet.name)
                self.t1_remove(packet)
                yield env.process(self.t2_append_left(env, res, packet))
            else:
                eff_lambda = self.adaptive_controller.compute_effective_lambda(
                    get_alpha(), self.history_manager.get_hp_stress()
                )
                self.t1_remove(packet)
                global_pos = round(len(self.t2) * eff_lambda)
                print("%s hit in t1, low priority, eff_lambda=%.3f, write to index %s in t2" % (
                    packet.name, eff_lambda, global_pos
                ))
                yield env.process(self.t2_append_by_index(env, res, packet, global_pos))

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s out of %s' % (self.tier.name, env.now))
            return

        # 2. Hit in T2
        if not is_write and self.t2.__contains__(packet.name):
            self.history_manager.record_access(
                packet.name, packet.priority, is_hit=True, hit_location="T2"
            )
            if packet.priority == "h":
                print("%s hit in t2, high priority, promote to MRU t2" % packet.name)
                self.t2_remove(packet)
                yield env.process(self.t2_append_left(env, res, packet))
            else:
                eff_lambda = self.adaptive_controller.compute_effective_lambda(
                    get_alpha(), self.history_manager.get_hp_stress()
                )
                eff_p = self.get_effective_p()
                current_pos = self.t2.__index__(packet.name)
                max_pos = max(0.0, self.c - eff_p)
                new_pos = int(min(max_pos, current_pos + round(len(self.t2) * eff_lambda)))
                new_pos = max(0, min(len(self.t2) - 1, new_pos)) if len(self.t2) > 1 else 0
                print("%s hit in t2, low priority, eff_lambda=%.3f, write to index %s in t2" % (
                    packet.name, eff_lambda, new_pos
                ))
                self.t2_remove(packet)
                yield env.process(self.t2_append_by_index(env, res, packet, new_pos))

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s out of %s' % (self.tier.name, env.now))
            return

        # If this was a read call and packet is neither in T1 nor T2, it's a read miss (no cache mutation)
        if not is_write:
            return

        # 3. Hit in Ghost list B1
        in_b1 = yield env.process(self.forwarder.index.packet_in_ghost(packet.name, 'b1'))
        if in_b1:
            self.history_manager.record_access(
                packet.name, packet.priority, is_hit=False, hit_location="B1"
            )
            len_b1 = yield env.process(self.forwarder.index.ghost_len('b1'))
            len_b2 = yield env.process(self.forwarder.index.ghost_len('b2'))
            self.increment_p(len_b1, len_b2)
            yield env.process(self._replace(env, packet))
            yield env.process(self.forwarder.index.del_packet_from_ghost(packet.name))
            if packet.priority == "h":
                print("%s hit in b1, high priority, promote to MRU t2" % packet.name)
                yield env.process(self.t2_append_left(env, res, packet))
            else:
                eff_lambda = self.adaptive_controller.compute_effective_lambda(
                    get_alpha(), self.history_manager.get_hp_stress()
                )
                global_pos = round(len(self.t2) * eff_lambda)
                print("%s hit in b1, low priority, eff_lambda=%.3f, write to index %s in t2" % (
                    packet.name, eff_lambda, global_pos
                ))
                yield env.process(self.t2_append_by_index(env, res, packet, global_pos))

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s out of %s' % (self.tier.name, env.now))
            return

        # 4. Hit in Ghost list B2
        in_b2 = yield env.process(self.forwarder.index.packet_in_ghost(packet.name, 'b2'))
        if in_b2:
            self.history_manager.record_access(
                packet.name, packet.priority, is_hit=False, hit_location="B2"
            )
            len_b1 = yield env.process(self.forwarder.index.ghost_len('b1'))
            len_b2 = yield env.process(self.forwarder.index.ghost_len('b2'))
            self.decrement_p(len_b1, len_b2)
            yield env.process(self._replace(env, packet))
            yield env.process(self.forwarder.index.del_packet_from_ghost(packet.name))
            if packet.priority == "h":
                print("%s hit in b2, high priority, promote to MRU t2" % packet.name)
                yield env.process(self.t2_append_left(env, res, packet))
            else:
                eff_lambda = self.adaptive_controller.compute_effective_lambda(
                    get_alpha(), self.history_manager.get_hp_stress()
                )
                global_pos = round(len(self.t2) * eff_lambda)
                print("%s hit in b2, low priority, eff_lambda=%.3f, write to index %s in t2" % (
                    packet.name, eff_lambda, global_pos
                ))
                yield env.process(self.t2_append_by_index(env, res, packet, global_pos))

            self.t1.__str__()
            self.t2.__str__()
            self.forwarder.index.__str__()
            self.forwarder.index.__str__(what="Ghost")
            print('%s out of %s' % (self.tier.name, env.now))
            return

        # 5. Cache Miss across all resident and ghost queues
        self.history_manager.record_access(
            packet.name, packet.priority, is_hit=False, hit_location="MISS"
        )
        len_b1 = yield env.process(self.forwarder.index.ghost_len('b1'))
        if len(self.t1) + len_b1 == self.c:
            print("%s cache miss in all queues (Case A)" % packet.name)
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

        # Check for active sequential scan streak
        is_scan = self.adaptive_controller.is_scan_active(self.history_manager.get_scan_score())

        if is_scan:
            # Sequential scan protection: insert at dampened low-tier position
            eff_lambda = self.adaptive_controller.compute_effective_lambda(
                get_alpha(), self.history_manager.get_hp_stress()
            )
            scan_pos = min(1, round(len(self.t1) * eff_lambda * 0.25))
            print("%s scan detected, write to dampened index %s in t1" % (packet.name, scan_pos))
            yield env.process(self.t1_append_by_index(env, res, packet, scan_pos))
        elif packet.priority == "h":
            print("%s high priority, write to MRU t1" % packet.name)
            yield env.process(self.t1_append_left(env, res, packet))
        else:
            eff_lambda = self.adaptive_controller.compute_effective_lambda(
                get_alpha(), self.history_manager.get_hp_stress()
            )
            global_pos = round(len(self.t1) * eff_lambda)
            print("%s low priority, eff_lambda=%.3f, write to index %s in t1" % (
                packet.name, eff_lambda, global_pos
            ))
            yield env.process(self.t1_append_by_index(env, res, packet, global_pos))

        self.t1.__str__()
        self.t2.__str__()
        self.forwarder.index.__str__()
        self.forwarder.index.__str__(what="Ghost")
        print('%s out of %s' % (self.tier.name, env.now))
        return
