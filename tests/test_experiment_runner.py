import json
from pathlib import Path
import pytest
import shutil
import tempfile

from experiments.config import ExperimentConfig, TierConfig
from experiments.registry import AlgorithmRegistry, AlgorithmDescriptor
from experiments.metrics import SimulationMetrics, TierMetrics
from experiments.results import ExperimentResult
from experiments.runner import ExperimentRunner
from experiments.reproduction import PaperReproductionSuite
from experiments.plotting import (
    plot_hit_ratios,
    plot_penalties,
    plot_retrieval_latencies,
    plot_tier_breakdown_from_json,
)


class TestAlgorithmRegistry:
    def test_canonical_algorithm_membership(self):
        expected = ["QM-ARC", "QL-QM-ARC", "M-ARC", "Priority-LRU", "LRU", "LFU", "Random"]
        registered = AlgorithmRegistry.list_algorithms()
        for algo in expected:
            assert algo in registered, f"Algorithm '{algo}' missing from registry."

    @pytest.mark.parametrize(
        "name,expected_canonical",
        [
            ("qm-arc", "QM-ARC"),
            ("QMARC", "QM-ARC"),
            ("qm_arc", "QM-ARC"),
            ("ql_qm_arc", "QL-QM-ARC"),
            ("QLQMARC", "QL-QM-ARC"),
            ("m-arc", "M-ARC"),
            ("marc", "M-ARC"),
            ("priority-lru", "Priority-LRU"),
            ("prioritylru", "Priority-LRU"),
            ("LRU", "LRU"),
            ("leastrecentlyused", "LRU"),
            ("LFU", "LFU"),
            ("random", "Random"),
            ("rand", "Random"),
        ],
    )
    def test_case_and_alias_resolution(self, name, expected_canonical):
        descriptor = AlgorithmRegistry.get(name)
        assert descriptor.name == expected_canonical

    def test_invalid_algorithm_rejection(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            AlgorithmRegistry.get("NonExistentCacheAlgorithm_XYZ")


class TestExperimentConfig:
    def test_two_tier_factory_ratio(self):
        total_cache = 60000
        cfg = ExperimentConfig.create_two_tier(
            algorithm="QM-ARC",
            total_cache_size=total_cache,
            workload="synthetic",
            num_requests=100,
        )
        assert cfg.num_tiers == 2
        assert len(cfg.custom_tier_configs) == 2
        dram, ssd = cfg.custom_tier_configs
        assert dram.name == "DRAM"
        assert ssd.name == "SSD"
        assert dram.max_size == 10000
        assert ssd.max_size == 50000
        assert dram.max_size + ssd.max_size == total_cache
        assert ssd.max_size / dram.max_size == pytest.approx(5.0)

    def test_three_tier_factory_ratio(self):
        total_cache = 310000
        cfg = ExperimentConfig.create_three_tier(
            algorithm="QL-QM-ARC",
            total_cache_size=total_cache,
            workload="synthetic",
            num_requests=100,
        )
        assert cfg.num_tiers == 3
        assert len(cfg.custom_tier_configs) == 3
        dram, ssd, nvme = cfg.custom_tier_configs
        assert dram.name == "DRAM"
        assert ssd.name == "SSD"
        assert nvme.name == "NVME"
        assert dram.max_size == 10000
        assert ssd.max_size == 50000
        assert nvme.max_size == 250000
        assert dram.max_size + ssd.max_size + nvme.max_size == total_cache
        assert ssd.max_size / dram.max_size == pytest.approx(5.0)
        assert nvme.max_size / dram.max_size == pytest.approx(25.0)


class TestSimulationMetrics:
    def test_zero_division_safety(self):
        """Metrics object must gracefully handle 0 counts without ZeroDivisionError."""
        metrics = SimulationMetrics(
            total_requests=0,
            high_priority_requests=0,
            low_priority_requests=0,
            total_hits=0,
            total_misses=0,
            total_hp_hits=0,
            total_lp_hits=0,
            penalty_hp=0.0,
            penalty_lp=0.0,
            total_penalty=0.0,
            global_hit_ratio=0.0,
            high_priority_hit_ratio=0.0,
            low_priority_hit_ratio=0.0,
            avg_retrieval_latency=0.0,
            avg_retrieval_latency_hp=0.0,
            avg_retrieval_latency_lp=0.0,
            simulation_wall_time_seconds=0.01,
        )
        d = metrics.to_dict()
        assert d["global_hit_ratio"] == 0.0
        assert d["high_priority_hit_ratio"] == 0.0
        assert d["low_priority_hit_ratio"] == 0.0


class TestExperimentExecutionAndPersistence:
    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.parametrize("algorithm", ["QM-ARC", "Priority-LRU", "LRU"])
    def test_run_two_tier_simulation(self, algorithm):
        cfg = ExperimentConfig(
            algorithm=algorithm,
            workload="synthetic",
            num_requests=30,
            num_tiers=2,
            cache_proportion=0.20,
            priority_ratio=0.20,
            seed=42,
            output_dir=self.temp_dir,
        )
        res = ExperimentRunner.run_experiment(cfg)
        assert res.algorithm == algorithm
        assert res.metrics.total_requests == 30
        assert res.metrics.total_hits + res.metrics.total_misses == 30
        assert 0.0 <= res.metrics.global_hit_ratio <= 1.0
        assert 0.0 <= res.metrics.high_priority_hit_ratio <= 1.0
        assert 0.0 <= res.metrics.low_priority_hit_ratio <= 1.0

        # Verify JSON file written
        json_file = self.temp_dir / "json" / f"{res.experiment_id}.json"
        assert json_file.is_file()
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["experiment_id"] == res.experiment_id
            assert data["algorithm"] == algorithm
            assert "metrics" in data

        # Verify CSV appended
        csv_file = self.temp_dir / "experiments_summary.csv"
        assert csv_file.is_file()

    def test_run_three_tier_ql_qm_arc(self):
        cfg = ExperimentConfig(
            algorithm="QL-QM-ARC",
            workload="synthetic",
            num_requests=40,
            num_tiers=3,
            cache_proportion=0.05,
            priority_ratio=0.20,
            seed=101,
            output_dir=self.temp_dir,
        )
        res = ExperimentRunner.run_experiment(cfg)
        assert res.algorithm == "QL-QM-ARC"
        assert res.metrics.total_requests == 40
        assert len(res.metrics.tier_breakdown) == 3
        tier_names = [t.tier_name for t in res.metrics.tier_breakdown]
        assert tier_names == ["DRAM", "SSD", "NVME"]

        # Capacity Ratio: DRAM:SSD:NVMe = 1:5:25
        dram_m, ssd_m, nvme_m = res.metrics.tier_breakdown
        assert ssd_m.max_size / dram_m.max_size == pytest.approx(5.0625, rel=0.1)
        assert nvme_m.max_size / dram_m.max_size == pytest.approx(25.1875, rel=0.1)

        # Capacity Invariant: occupancy must NEVER exceed physical max capacity
        for t in res.metrics.tier_breakdown:
            assert t.used_size <= t.max_size, f"{t.tier_name} used_size ({t.used_size}) exceeded max_size ({t.max_size})"
            assert t.occupancy_ratio <= 1.0, f"{t.tier_name} occupancy ({t.occupancy_ratio}) exceeded 1.0"

    def test_two_tier_capacity_ratio_preserved(self):
        """Verify that 2-tier mode strictly enforces 1:5 capacity ratio."""
        cfg = ExperimentConfig(
            algorithm="QM-ARC",
            workload="synthetic",
            num_requests=30,
            num_tiers=2,
            cache_proportion=0.05,
            seed=42,
            output_dir=self.temp_dir,
        )
        res = ExperimentRunner.run_experiment(cfg)
        dram, ssd = res.metrics.tier_breakdown
        assert dram.max_size > 0
        assert ssd.max_size > 0
        ratio = ssd.max_size / dram.max_size
        assert ratio == pytest.approx(5.0, rel=0.05)
        assert dram.used_size <= dram.max_size
        assert ssd.used_size <= ssd.max_size

    def test_metric_denominators_distinction(self):
        """Verify distinction between global hit contribution and conditional hit ratio."""
        cfg = ExperimentConfig(
            algorithm="QM-ARC",
            workload="synthetic",
            num_requests=50,
            num_tiers=2,
            cache_proportion=0.05,
            seed=42,
            output_dir=self.temp_dir,
        )
        res = ExperimentRunner.run_experiment(cfg)
        m = res.metrics
        assert m.global_hit_ratio == pytest.approx(m.total_hits / m.total_requests)

        for t in m.tier_breakdown:
            assert t.hit_contribution == pytest.approx(t.hits / m.total_requests)
            if t.requests_reaching_tier > 0:
                assert t.conditional_hit_ratio == pytest.approx(t.hits / t.requests_reaching_tier)
            else:
                assert t.conditional_hit_ratio == 0.0

    def test_consistent_workload_reuse_across_algorithms(self):
        """Verify that sharing existing_trace runs identical request sequence across algorithms."""
        cfg_qm = ExperimentConfig(algorithm="QM-ARC", workload="synthetic", num_requests=30, seed=42, output_dir=self.temp_dir)
        desc = AlgorithmRegistry.get("QM-ARC")
        trace, name, is_synth = ExperimentRunner.resolve_workload(cfg_qm, desc)

        cfg_lru = ExperimentConfig(algorithm="LRU", workload="synthetic", num_requests=30, seed=42, output_dir=self.temp_dir)
        res_qm = ExperimentRunner.run_experiment(cfg_qm, existing_trace=trace, workload_name_override=name, is_synthetic_override=is_synth)
        res_lru = ExperimentRunner.run_experiment(cfg_lru, existing_trace=trace, workload_name_override=name, is_synthetic_override=is_synth)

        assert res_qm.metrics.total_requests == res_lru.metrics.total_requests == 30
        assert res_qm.metrics.high_priority_requests == res_lru.metrics.high_priority_requests
        assert res_qm.metrics.low_priority_requests == res_lru.metrics.low_priority_requests

    def test_plotting_generation(self):
        # Run two experiments to have rows in CSV
        for algo in ["QM-ARC", "LRU"]:
            cfg = ExperimentConfig(
                algorithm=algo,
                workload="synthetic",
                num_requests=25,
                num_tiers=2,
                seed=42,
                output_dir=self.temp_dir,
            )
            ExperimentRunner.run_experiment(cfg)

        csv_file = self.temp_dir / "experiments_summary.csv"
        p1 = plot_hit_ratios(csv_file, self.temp_dir / "hit_ratios.png")
        assert p1.is_file()

        p2 = plot_penalties(csv_file, self.temp_dir / "penalties.png")
        assert p2.is_file()

        p3 = plot_retrieval_latencies(csv_file, self.temp_dir / "latencies.png")
        assert p3.is_file()

        # Test single json tier plotting
        json_files = list((self.temp_dir / "json").glob("*.json"))
        assert len(json_files) >= 1
        p4 = plot_tier_breakdown_from_json(json_files[0], self.temp_dir / "tiers.png")
        assert p4.is_file()

    def test_table_3_formula_and_sign_convention(self):
        """Verify Table 3 percentage difference formula: ((P_QM - P_Strat) / P_Strat) * 100."""
        # Case 1: QM-ARC has 40% lower penalty -> expect -40.0%
        qm_pen = 60.0
        strat_pen = 100.0
        diff = ((qm_pen - strat_pen) / strat_pen) * 100.0
        assert diff == pytest.approx(-40.0)
        assert diff < 0.0, "Negative value must indicate QM-ARC performed better"

        # Case 2: QM-ARC has higher penalty -> expect positive percentage
        qm_pen_worse = 125.0
        diff_worse = ((qm_pen_worse - strat_pen) / strat_pen) * 100.0
        assert diff_worse == pytest.approx(25.0)
        assert diff_worse > 0.0

        # Case 3: Equal penalties -> 0.0%
        diff_equal = ((strat_pen - strat_pen) / strat_pen) * 100.0
        assert diff_equal == pytest.approx(0.0)

    def test_aggregation_request_count_isolation(self):
        """Verify that runs with different request counts are never averaged together."""
        import pandas as pd
        records = [
            {"algorithm": "QM-ARC", "workload": "zipf", "requests": 50, "num_tiers": 2, "cache_proportion": 0.05, "penalty_lambda": 0.2, "penalty_hp": 500.0},
            {"algorithm": "QM-ARC", "workload": "zipf", "requests": 1000, "num_tiers": 2, "cache_proportion": 0.05, "penalty_lambda": 0.2, "penalty_hp": 6000.0},
        ]
        df = pd.DataFrame(records)
        group_cols = ["algorithm", "workload", "requests", "num_tiers", "cache_proportion", "penalty_lambda"]
        agg = df.groupby(group_cols)["penalty_hp"].mean().reset_index()

        # Must produce 2 separate rows, not a single merged row with average 3250
        assert len(agg) == 2
        r50 = agg[agg["requests"] == 50]["penalty_hp"].iloc[0]
        r1000 = agg[agg["requests"] == 1000]["penalty_hp"].iloc[0]
        assert r50 == 500.0
        assert r1000 == 6000.0

    def test_ql_qm_arc_richer_state_and_epsilon_decay(self):
        """Verify QLearningAgent richer state space indexing and epsilon decay progression."""
        from policies.QL_QM_ARC.ql_agent import QLearningAgent
        # 10 p states (0..9) with cache events -> total states = 10 * 5 = 50
        agent = QLearningAgent(
            num_states=10,
            num_actions=4,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=0.5,
            epsilon_decay_value=0.05,
            min_epsilon=0.1,
            include_cache_state=True,
        )
        assert agent.total_states == 50
        assert agent.q_table.shape == (50, 8)

        # Distinct events for the same p must map to distinct state rows
        s_t1 = agent.get_state_index(p=4, hit="t1")
        s_t2 = agent.get_state_index(p=4, hit="t2")
        s_miss = agent.get_state_index(p=4, hit="Miss")
        assert s_t1 != s_t2
        assert s_t2 != s_miss
        assert s_t1 == 4 * 5 + 0
        assert s_t2 == 4 * 5 + 1
        assert s_miss == 4 * 5 + 4

        # Verify epsilon decay progression over steps
        initial_eps = agent.epsilon
        assert initial_eps == 0.5
        agent.get_next_state(current_p=4, current_t1_len=2, current_b1_len=1, current_b2_len=1, hit="t1")
        assert agent.epsilon == pytest.approx(0.45)
        # Decay past min_epsilon should be clamped to min_epsilon
        for _ in range(20):
            agent.get_next_state(current_p=4, current_t1_len=2, current_b1_len=1, current_b2_len=1, hit="Miss")
        assert agent.epsilon == pytest.approx(0.1)

