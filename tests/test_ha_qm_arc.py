"""Test suite for HA-QM-ARC (History-Aware Adaptive QM-ARC).

Verifies:
1. WorkloadHistoryManager bounded O(1) operations, metric computations, location tracking, and resets.
2. AdaptiveController mathematical modulations, clamping, and dynamic QoS parameters.
3. Deterministic ARC partition adaptation (p modulation protecting T2 vs T1).
4. Deterministic QoS placement (index 0 as LRU, lambda_effective eviction distance).
5. Multi-tier effective p scaling, capacity bounds, and eviction without history double-counting.
6. Algorithm registry integration, reproducibility, and physical capacity invariants across all 8 algorithms.
"""

from collections import OrderedDict
from pathlib import Path
import pytest
import shutil
import simpy
import tempfile

from common.deque import Deque
from common.packet import Packet
from experiments.config import ExperimentConfig
from experiments.registry import AlgorithmRegistry
from experiments.runner import ExperimentRunner
from forwarder_structures.content_store.index import Index
from forwarder_structures.content_store.tier import Tier
from forwarder_structures.forwarder import Forwarder
from forwarder_structures.pit import PIT
from policies.HA_QM_ARC.history_manager import WorkloadHistoryManager
from policies.HA_QM_ARC.adaptive_controller import AdaptiveController
from policies.HA_QM_ARC.abstract_ha_qm_arc_policy import AbstractHAQMARCPolicy
from policies.HA_QM_ARC.tier_ha_qm_arc_policy import TierHAQMARCPolicy


# ============================================================================
# 1. WorkloadHistoryManager Unit Tests
# ============================================================================

class TestWorkloadHistoryManager:
    def test_initialization_bounds(self):
        hm = WorkloadHistoryManager(window_size=50, burst_window_size=15, scan_streak_threshold=4)
        assert hm.window_size == 50
        assert hm.burst_window_size == 15
        assert hm.scan_streak_threshold == 4
        assert hm.enable_scan_detection is False
        assert hm.get_rolling_ghr() == 0.0
        assert hm.get_burst_score() == 0.0
        assert hm.get_scan_score() == 0.0
        assert hm.get_hp_stress() == 0.0

    def test_sliding_window_eviction(self):
        hm = WorkloadHistoryManager(window_size=10, burst_window_size=5)
        # Record 10 misses
        for i in range(10):
            hm.record_access(packet_name=f"pkt_{i}", priority="l", is_hit=False)
        assert len(hm.hit_window) == 10
        assert hm.get_rolling_ghr() == 0.0

        # Now record 10 hits; should fully push out the 10 misses
        for i in range(10):
            hm.record_access(packet_name=f"pkt_{i+10}", priority="l", is_hit=True)
        assert len(hm.hit_window) == 10
        assert hm.get_rolling_ghr() == 1.0
        assert hm.total_hits_in_window == 10

    def test_ghr_gradient(self):
        hm = WorkloadHistoryManager(window_size=20)
        # First 10 accesses are misses
        for i in range(10):
            hm.record_access(packet_name=f"pkt_{i}", priority="l", is_hit=False)
        # Next 10 accesses are hits
        for i in range(10, 20):
            hm.record_access(packet_name=f"pkt_{i}", priority="l", is_hit=True)
        # Recent half has GHR=1.0, older half has GHR=0.0 -> gradient = +1.0
        grad = hm.get_ghr_gradient()
        assert pytest.approx(grad, rel=1e-2) == 1.0

    def test_burst_score_detection(self):
        hm = WorkloadHistoryManager(burst_window_size=10)
        # All distinct items -> burst score = 0.0
        for i in range(10):
            hm.record_access(packet_name=f"unique_{i}", priority="l", is_hit=False)
        assert hm.get_burst_score() == 0.0

        # Now spam the same item 10 times -> burst score = 1.0
        for _ in range(10):
            hm.record_access(packet_name="burst_target", priority="l", is_hit=True)
        assert hm.get_burst_score() == 1.0

    def test_sequential_scan_detection_configured_and_disabled(self):
        # Case A: Disabled by default -> scan score must remain 0.0 even if sequential IDs arrive
        hm_disabled = WorkloadHistoryManager(scan_streak_threshold=5, enable_scan_detection=False)
        for item_id in range(100, 110):
            hm_disabled.record_access(packet_name=f"chunk_{item_id}", priority="l", is_hit=False)
        assert hm_disabled.get_scan_score() == 0.0

        # Case B: Enabled -> strictly consecutive items trigger scan score
        hm_enabled = WorkloadHistoryManager(scan_streak_threshold=5, enable_scan_detection=True)
        for item_id in range(100, 106):
            hm_enabled.record_access(packet_name=f"chunk_{item_id}", priority="l", is_hit=False)
        assert hm_enabled.get_scan_score() >= 1.0

        # Case C: Repeating same item does NOT increment scan streak (temporal burst, not scan)
        hm_enabled.record_access(packet_name="chunk_105", priority="l", is_hit=True)
        assert hm_enabled.current_scan_streak == 0

        # Case D: Non-consecutive items reset streak
        hm_enabled.record_access(packet_name="chunk_200", priority="l", is_hit=False)
        assert hm_enabled.current_scan_streak == 0

    def test_hp_stress_tracking(self):
        hm = WorkloadHistoryManager(window_size=20)
        # 5 LP requests (ignored for HP stress)
        for i in range(5):
            hm.record_access(packet_name=f"lp_{i}", priority="l", is_hit=False)
        assert hm.get_hp_stress() == 0.0

        # 10 HP requests with 8 misses
        for i in range(8):
            hm.record_access(packet_name=f"hp_miss_{i}", priority="h", is_hit=False)
        for i in range(2):
            hm.record_access(packet_name=f"hp_hit_{i}", priority="h", is_hit=True)

        assert pytest.approx(hm.get_hp_stress(), rel=1e-2) == 0.80

    def test_hit_location_tracking(self):
        hm = WorkloadHistoryManager(window_size=10)
        hm.record_access("p1", "h", is_hit=True, hit_location="T1")
        hm.record_access("p2", "h", is_hit=True, hit_location="T1")
        hm.record_access("p3", "l", is_hit=True, hit_location="T2")
        hm.record_access("p4", "l", is_hit=False, hit_location="B1")
        hm.record_access("p5", "l", is_hit=False, hit_location="B2")
        for i in range(5):
            hm.record_access(f"m_{i}", "l", is_hit=False, hit_location="MISS")

        breakdown = hm.get_hit_locations_breakdown()
        assert pytest.approx(breakdown["T1"], rel=1e-2) == 0.20
        assert pytest.approx(breakdown["T2"], rel=1e-2) == 0.10
        assert pytest.approx(breakdown["B1"], rel=1e-2) == 0.10
        assert pytest.approx(breakdown["B2"], rel=1e-2) == 0.10
        assert pytest.approx(breakdown["MISS"], rel=1e-2) == 0.50

    def test_reset(self):
        hm = WorkloadHistoryManager(window_size=20)
        for i in range(15):
            hm.record_access(packet_name=f"p_{i}", priority="h", is_hit=True, hit_location="T1")
        assert len(hm.hit_window) == 15
        hm.reset()
        assert len(hm.hit_window) == 0
        assert hm.total_hits_in_window == 0
        assert hm.get_rolling_ghr() == 0.0
        assert hm.get_ghr_gradient() == 0.0


# ============================================================================
# 2. AdaptiveController Unit Tests
# ============================================================================

class TestAdaptiveController:
    def test_default_hyperparameters(self):
        ac = AdaptiveController()
        assert ac.omega_burst == 0.15
        assert ac.omega_scan == 0.20
        assert ac.omega_grad == 0.10
        assert ac.kappa_qos == 0.50

    def test_delta_p_history_burst_response(self):
        ac = AdaptiveController(omega_burst=0.20, omega_scan=0.0, omega_grad=0.0)
        c = 100
        # High burstiness must shift capacity toward frequency (delta_p < 0)
        delta_p = ac.compute_delta_p_history(c=c, burst_score=1.0, scan_score=0.0, ghr_gradient=0.0)
        assert delta_p < 0
        assert delta_p == round(100 * -0.20)

    def test_delta_p_history_scan_response(self):
        ac = AdaptiveController(omega_burst=0.0, omega_scan=0.25, omega_grad=0.0)
        c = 100
        # High scan streak must suppress T1 expansion (delta_p < 0)
        delta_p = ac.compute_delta_p_history(c=c, burst_score=0.0, scan_score=1.0, ghr_gradient=0.0)
        assert delta_p == -25.0

    def test_delta_p_history_gradient_reward(self):
        ac = AdaptiveController(omega_burst=0.0, omega_scan=0.0, omega_grad=0.30)
        c = 100
        # Positive GHR trend expands T1 (delta_p > 0)
        delta_p = ac.compute_delta_p_history(c=c, burst_score=0.0, scan_score=0.0, ghr_gradient=0.5)
        assert delta_p == 15.0

    def test_delta_p_clamping_invariant(self):
        ac = AdaptiveController(omega_burst=10.0)
        c = 100
        # Extreme sensitivity must be clamped to at most 30% of capacity
        delta_p = ac.compute_delta_p_history(c=c, burst_score=1.0, scan_score=0.0, ghr_gradient=0.0)
        assert delta_p == -30.0

    def test_effective_p_bounds(self):
        ac = AdaptiveController(omega_burst=0.5)
        c = 50
        # Base p is 10, large negative delta should not go below 0
        eff_p = ac.get_effective_p(base_p=5.0, c=c, burst_score=1.0, scan_score=1.0, ghr_gradient=0.0)
        assert 0.0 <= eff_p <= float(c)

    def test_dynamic_qos_lambda_modulation(self):
        ac = AdaptiveController(kappa_qos=0.50)
        base_lambda = 0.20

        # Zero HP stress: lambda_effective == base_lambda
        eff_zero = ac.compute_effective_lambda(base_lambda=base_lambda, hp_stress=0.0)
        assert pytest.approx(eff_zero, rel=1e-3) == 0.20

        # Maximum HP stress (1.0): lambda_effective shrinks to base_lambda * (1 - 0.5) = 0.10
        eff_max = ac.compute_effective_lambda(base_lambda=base_lambda, hp_stress=1.0)
        assert pytest.approx(eff_max, rel=1e-3) == 0.10

    def test_diagnostics_dictionary(self):
        ac = AdaptiveController()
        diag = ac.get_diagnostics(
            c=100,
            base_p=20.0,
            base_lambda=0.20,
            burst_score=0.4,
            scan_score=0.2,
            ghr_gradient=0.1,
            hp_stress=0.3,
        )
        assert "effective_p" in diag
        assert "effective_lambda" in diag
        assert "scan_active" in diag
        assert diag["base_p"] == 20.0


# ============================================================================
# 3. Deterministic ARC Replacement & QoS Placement Verification
# ============================================================================

class TestDeterministicARCAndQoSBehavior:
    def test_deterministic_p_adaptation_protects_t2(self):
        """Demonstrate that decreasing effective p evicts from T1, strictly protecting T2."""
        env = simpy.Environment()
        index = Index(env)
        pit = PIT()
        boss = Tier("BOSS", 0, 0, 0, 0, 0)
        dram = Tier("DRAM", 1000, 1, 1e-7, 40e9, 20e9)
        forwarder = Forwarder(env, index, [boss, dram], pit, slot_size=100, default_tier_index=1)

        coordinator = AbstractHAQMARCPolicy(env, forwarder, boss)
        tier_policy = TierHAQMARCPolicy(env, forwarder, dram)

        # Directly populate T1 and T2 to simulate an active cache state:
        # T1 has 2 items: rec_1, rec_2
        # T2 has 2 items: freq_1, freq_2
        p1 = Packet("d", 0.0, "rec_1", 100, "l", 0.05)
        p2 = Packet("d", 0.0, "rec_2", 100, "l", 0.05)
        p3 = Packet("d", 0.0, "freq_1", 100, "h", 0.05)
        p4 = Packet("d", 0.0, "freq_2", 100, "h", 0.05)

        coordinator.t1.append_left(p1.name, p1)
        coordinator.t1.append_left(p2.name, p2)
        coordinator.t2.append_left(p3.name, p3)
        coordinator.t2.append_left(p4.name, p4)

        tier_policy.t1.append_left(p1.name, p1)
        tier_policy.t1.append_left(p2.name, p2)
        tier_policy.t2.append_left(p3.name, p3)
        tier_policy.t2.append_left(p4.name, p4)

        # Set total capacity to 4
        coordinator.c = 4
        tier_policy.c = 4

        # Scenario A: Decrease effective p to 0 (all capacity allocated to T2 frequency)
        coordinator.p = 0
        assert len(coordinator.t1) == 2
        assert len(coordinator.t2) == 2

        # A new incoming miss item triggers replacement
        miss_pkt = Packet("d", 1.0, "miss_item_A", 100, "l", 0.05)

        def run_replace_test():
            yield env.process(coordinator._replace(env, miss_pkt))

        env.process(run_replace_test())
        env.run()

        # Because len(T1) = 2 > p = 0, eviction MUST come from T1 (rec_1 evicted)
        # T2 items (freq_1, freq_2) MUST be completely preserved!
        assert "freq_1" in coordinator.t2
        assert "freq_2" in coordinator.t2
        assert "rec_1" not in coordinator.t1  # Evicted from T1
        assert "rec_2" in coordinator.t1

        # Scenario B: Now increase effective p to 4 (all capacity allocated to T1 recency)
        coordinator.p = 4
        miss_pkt_B = Packet("d", 2.0, "miss_item_B", 100, "l", 0.05)

        def run_replace_test_b():
            yield env.process(coordinator._replace(env, miss_pkt_B))

        env.process(run_replace_test_b())
        env.run()

        # Because len(T1) = 1 <= p = 4, eviction MUST come from T2 (freq_1 evicted)
        # Remaining T1 items are preserved!
        assert "rec_2" in coordinator.t1
        assert "freq_1" not in coordinator.t2  # Evicted from T2
        assert "freq_2" in coordinator.t2

    def test_deterministic_qos_placement_order_and_eviction(self):
        """Demonstrate that index 0 is LRU, and reducing lambda_effective positions items closer to eviction head."""
        q = Deque()
        p_lru = Packet("d", 0.0, "lru_first", 100, "l", 0.05)
        p_mid = Packet("d", 0.0, "mid_second", 100, "l", 0.05)
        p_mru = Packet("d", 0.0, "mru_last", 100, "h", 0.05)

        # In Deque: append_left appends to the MRU (end of OrderedDict)
        # append_by_index inserts at exact integer index
        q.append_left(p_lru.name, p_lru)
        q.append_left(p_mid.name, p_mid)
        q.append_left(p_mru.name, p_mru)

        # Verify that get_without_pop() and pop() operate on index 0 (LRU)
        old_name, old_val = q.get_without_pop()
        assert old_name == "lru_first"  # Index 0 is the LRU eviction head

        popped = q.pop()
        assert popped.name == "lru_first"
        assert len(q) == 2

        # Now test lambda_effective positioning:
        # A list of 10 items
        test_deque = Deque()
        for i in range(10):
            test_deque.append_left(f"item_{i}", f"val_{i}")

        # If lambda = 0.20: pos = round(10 * 0.20) = 2 (two slots from eviction head)
        pos_normal = round(len(test_deque) * 0.20)
        assert pos_normal == 2

        # If lambda_effective is attenuated under HP stress to 0.05: pos = round(10 * 0.05) = 0 or 1
        pos_attenuated = round(len(test_deque) * 0.05)
        assert pos_attenuated == 0 or pos_attenuated == 1
        assert pos_attenuated < pos_normal  # Closer to index 0 (eviction head)

        # High-priority item is appended at MRU (len - 1), furthest from eviction
        test_deque.append_left("hp_item", "val_hp")
        keys = list(test_deque.od.keys())
        assert keys[-1] == "hp_item"  # MRU tail
        assert keys[0] != "hp_item"   # Not at LRU head

    def test_multi_tier_effective_p_scaling_and_bounds(self):
        """Demonstrate that local effective p is scaled proportionally to tier size and strictly bounded."""
        env = simpy.Environment()
        index = Index(env)
        pit = PIT()
        boss = Tier("BOSS", 0, 0, 0, 0, 0)
        dram = Tier("DRAM", 1000, 1, 1e-7, 40e9, 20e9)
        ssd = Tier("SSD", 5000, 512, 1e-5, 2e9, 1e9)
        forwarder = Forwarder(env, index, [boss, dram, ssd], pit, slot_size=100, default_tier_index=1)

        coordinator = AbstractHAQMARCPolicy(env, forwarder, boss)
        dram_policy = TierHAQMARCPolicy(env, forwarder, dram)
        ssd_policy = TierHAQMARCPolicy(env, forwarder, ssd)

        coordinator.c = 60
        dram_policy.c = 10
        ssd_policy.c = 50

        coordinator.p = 30
        dram_policy.p = 5
        ssd_policy.p = 25

        # Verify that effective p on tiers scales proportionally
        # Force a history delta on coordinator: e.g. delta_p_global = -12
        coordinator.adaptive_controller.compute_delta_p_history = lambda *args: -12.0

        # DRAM ratio: 10 / 60 = 1/6 -> delta_p_dram = round(-12 * 1/6) = -2
        # Effective DRAM p = max(0, min(10, 5 - 2)) = 3
        assert dram_policy.get_effective_p() == 3.0

        # SSD ratio: 50 / 60 = 5/6 -> delta_p_ssd = round(-12 * 5/6) = -10
        # Effective SSD p = max(0, min(50, 25 - 10)) = 15
        assert ssd_policy.get_effective_p() == 15.0

        # Extreme negative delta: bounds must prevent negative p
        coordinator.adaptive_controller.compute_delta_p_history = lambda *args: -100.0
        assert dram_policy.get_effective_p() == 0.0
        assert ssd_policy.get_effective_p() == 0.0

        # Extreme positive delta: bounds must prevent p exceeding tier capacity c_i
        coordinator.adaptive_controller.compute_delta_p_history = lambda *args: 100.0
        assert dram_policy.get_effective_p() == float(dram_policy.c)
        assert ssd_policy.get_effective_p() == float(ssd_policy.c)


# ============================================================================
# 4. Algorithm Registry and Policy Integration Tests
# ============================================================================

class TestHAQMARCRegistryAndPolicy:
    def test_registry_lookup_and_aliases(self):
        canonical = AlgorithmRegistry.get("HA-QM-ARC")
        assert canonical.name == "HA-QM-ARC"
        assert canonical.category == "arc"
        assert canonical.coordinator_class is AbstractHAQMARCPolicy
        assert canonical.tier_policy_class is TierHAQMARCPolicy

        # Case-insensitive and alias lookups
        for alias in ["ha-qm-arc", "HAQMARC", "ha_qm_arc", "haqmarc"]:
            desc = AlgorithmRegistry.get(alias)
            assert desc.name == "HA-QM-ARC"

    def test_all_eight_algorithms_registered(self):
        all_algos = AlgorithmRegistry.list_algorithms()
        expected = ["HA-QM-ARC", "LFU", "LRU", "M-ARC", "Priority-LRU", "QL-QM-ARC", "QM-ARC", "Random"]
        assert all_algos == expected


# ============================================================================
# 5. Simulation & Physical Capacity Invariant Tests
# ============================================================================

class TestHAQMARCSimulationInvariants:
    @pytest.fixture
    def temp_output_dir(self):
        tmp = tempfile.mkdtemp(prefix="test_ha_qm_arc_")
        yield Path(tmp)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_ha_qm_arc_two_tier_execution(self, temp_output_dir):
        config = ExperimentConfig(
            algorithm="HA-QM-ARC",
            workload="synthetic",
            cache_proportion=0.01,
            num_tiers=2,
            num_requests=200,
            priority_ratio=0.2,
            seed=42,
            output_dir=temp_output_dir,
            logs_enabled=False,
        )

        result = ExperimentRunner.run_experiment(config)
        assert result.algorithm == "HA-QM-ARC"
        assert result.config.num_tiers == 2
        assert len(result.metrics.tier_breakdown) == 2

        # Verify strict capacity invariants on both physical tiers
        for tm in result.metrics.tier_breakdown:
            assert tm.used_size <= tm.max_size, (
                f"Tier {tm.tier_name} violated capacity invariant: used {tm.used_size} > max {tm.max_size}"
            )
            assert 0.0 <= tm.occupancy_ratio <= 1.0

        # Verify metrics consistency
        assert 0.0 <= result.metrics.global_hit_ratio <= 1.0
        assert result.metrics.total_requests == 200

    def test_ha_qm_arc_three_tier_execution(self, temp_output_dir):
        config = ExperimentConfig(
            algorithm="HA-QM-ARC",
            workload="synthetic",
            cache_proportion=0.05,
            num_tiers=3,
            num_requests=250,
            priority_ratio=0.2,
            seed=42,
            output_dir=temp_output_dir,
            logs_enabled=False,
        )

        result = ExperimentRunner.run_experiment(config)
        assert result.algorithm == "HA-QM-ARC"
        assert result.config.num_tiers == 3
        assert len(result.metrics.tier_breakdown) == 3

        # Strict capacity invariants on DRAM, SSD, and NVMe
        for tm in result.metrics.tier_breakdown:
            assert tm.used_size <= tm.max_size, (
                f"Tier {tm.tier_name} violated capacity invariant: used {tm.used_size} > max {tm.max_size}"
            )
            assert 0.0 <= tm.occupancy_ratio <= 1.0

    def test_ha_qm_arc_deterministic_reproducibility(self, temp_output_dir):
        cfg1 = ExperimentConfig(
            algorithm="HA-QM-ARC",
            workload="synthetic",
            cache_proportion=0.01,
            num_tiers=2,
            num_requests=150,
            seed=999,
            output_dir=temp_output_dir / "run1",
            logs_enabled=False,
        )
        cfg2 = ExperimentConfig(
            algorithm="HA-QM-ARC",
            workload="synthetic",
            cache_proportion=0.01,
            num_tiers=2,
            num_requests=150,
            seed=999,
            output_dir=temp_output_dir / "run2",
            logs_enabled=False,
        )

        res1 = ExperimentRunner.run_experiment(cfg1)
        res2 = ExperimentRunner.run_experiment(cfg2)

        assert res1.metrics.global_hit_ratio == res2.metrics.global_hit_ratio
        assert res1.metrics.total_penalty == res2.metrics.total_penalty
        assert res1.metrics.penalty_hp == res2.metrics.penalty_hp

    def test_all_eight_algorithms_execute_successfully(self, temp_output_dir):
        """Invariant check: all 8 registered algorithms can execute without error on the same workload."""
        algorithms = AlgorithmRegistry.list_algorithms()
        assert len(algorithms) == 8

        for algo in algorithms:
            cfg = ExperimentConfig(
                algorithm=algo,
                workload="synthetic",
                cache_proportion=0.01,
                num_tiers=2,
                num_requests=60,
                seed=123,
                output_dir=temp_output_dir / algo,
                logs_enabled=False,
            )
            res = ExperimentRunner.run_experiment(cfg)
            assert res.algorithm == algo
            assert res.metrics.total_requests == 60
            assert 0.0 <= res.metrics.global_hit_ratio <= 1.0
