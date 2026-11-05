import os
from pathlib import Path
import random
import time
from typing import Tuple, List, Optional, Union

import numpy as np
import simpy

import resources
from common.packet import Packet
from common.penalty import set_penalty_lambda
from forwarder_structures.content_store.index import Index
from forwarder_structures.content_store.tier import Tier
from forwarder_structures.forwarder import Forwarder
from forwarder_structures.pit import PIT
from simulation import Simulation
from traces.trace_creating_and_parsing.synthetic_trace import generate_synthetic_trace
from traces.trace_creating_and_parsing.jedi_trace import parse_raw_jedi_trace, generate_synthetic_jedi_workload
from traces.trace_creating_and_parsing.snia_trace import parse_ibm_snia_trace
from traces.trace_reading.trace import Trace

from experiments.config import ExperimentConfig, TierConfig
from experiments.metrics import SimulationMetrics
from experiments.registry import AlgorithmRegistry, AlgorithmDescriptor
from experiments.results import ExperimentResult


class ExperimentRunner:
    """Configurable runner executing reproducible multi-tier cache simulations."""

    @staticmethod
    def resolve_workload(config: ExperimentConfig, descriptor: AlgorithmDescriptor) -> Tuple[Trace, str, bool]:
        """Resolve, generate, or load trace data according to the experiment configuration.

        Returns:
            trace_instance: Loaded Trace object ready for simulation.
            workload_name: Human-readable workload identifier.
            is_synthetic: Flag indicating whether trace is synthetic or authentic.
        """
        # Ensure reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        workload_str = config.workload.strip().lower()
        is_synthetic = True
        workload_path: Optional[Path] = None
        workload_name = config.workload

        if Path(config.workload).is_file():
            workload_path = Path(config.workload)
            workload_name = workload_path.stem
            is_synthetic = "synthetic" in workload_path.name.lower() or "approx" in workload_path.name.lower()

        elif workload_str in ("synthetic", "zipf"):
            workload_name = f"synthetic_zipf_alpha1.0_hp{config.priority_ratio}"
            filename = f"synth_items10000_reqs{config.num_requests}_seed{config.seed}.csv"
            workload_path = resources.SYNTHETIC_DIR / filename
            if not workload_path.is_file():
                generate_synthetic_trace(
                    n_unique_items=10000,
                    zipf_alpha=1.0,
                    high_priority_ratio=config.priority_ratio,
                    num_requests=config.num_requests,
                    seed=config.seed,
                    output_path=workload_path,
                )
            is_synthetic = True

        elif workload_str in ("jedi", "video"):
            raw_jedi = resources.RAW_DATASET_DIR / "jedi_w"
            if raw_jedi.is_file():
                workload_path = parse_raw_jedi_trace(
                    raw_input_path=raw_jedi,
                    high_priority_ratio=config.priority_ratio,
                    trace_len_limit=config.num_requests,
                    seed=config.seed,
                )
                workload_name = "jedi_video_authentic"
                is_synthetic = False
            else:
                filename = f"jedi_video_approx_items40000_reqs{config.num_requests}_seed{config.seed}.csv"
                workload_path = resources.JEDI_DIR / filename
                if not workload_path.is_file():
                    generate_synthetic_jedi_workload(
                        n_unique_items=40000,
                        num_requests=config.num_requests,
                        high_priority_ratio=config.priority_ratio,
                        seed=config.seed,
                        output_path=workload_path,
                    )
                workload_name = "jedi_video_synthetic_approx"
                is_synthetic = True

        elif workload_str in ("snia", "ibm"):
            raw_ibm = resources.RAW_DATASET_DIR / "IBMObjectStoreTrace001Part0"
            if raw_ibm.is_file():
                workload_path = parse_ibm_snia_trace(
                    raw_input_path=raw_ibm,
                    high_priority_ratio=config.priority_ratio,
                    trace_len_limit=config.num_requests,
                    seed=config.seed,
                )
                workload_name = "ibm_object_store_authentic"
                is_synthetic = False
            else:
                sample_raw = resources.RAW_DATASET_DIR / "sample_IBMObjectStoreTrace001Part0.txt"
                workload_path = parse_ibm_snia_trace(
                    raw_input_path=sample_raw,
                    high_priority_ratio=config.priority_ratio,
                    trace_len_limit=config.num_requests,
                    seed=config.seed,
                )
                workload_name = "ibm_object_store_fixture"
                is_synthetic = True

        else:
            raise FileNotFoundError(f"Workload '{config.workload}' could not be resolved.")

        # Instantiate the trace reader class bound to this algorithm
        trace_instance = descriptor.trace_reader_class()
        # Temporarily redirect NDN_PACKETS to the single resolved file
        old_packets = list(resources.NDN_PACKETS)
        resources.NDN_PACKETS = [str(workload_path)]
        try:
            trace_instance.gen_data(trace_len_limit=config.num_requests, file_path=workload_path)
        finally:
            resources.NDN_PACKETS = old_packets

        return trace_instance, workload_name, is_synthetic

    @staticmethod
    def run_experiment(
        config: ExperimentConfig,
        existing_trace: Optional[Trace] = None,
        workload_name_override: Optional[str] = None,
        is_synthetic_override: Optional[bool] = None,
    ) -> ExperimentResult:
        """Execute a complete cache simulation according to ExperimentConfig."""
        descriptor = AlgorithmRegistry.get(config.algorithm)

        # Set the QoS penalty lambda parameter
        set_penalty_lambda(config.penalty_lambda)

        if existing_trace is not None:
            if isinstance(existing_trace, descriptor.trace_reader_class):
                trace = existing_trace
            else:
                trace = descriptor.trace_reader_class()
                trace.data = list(existing_trace.data)
            workload_name = workload_name_override or config.workload
            is_synthetic = is_synthetic_override if is_synthetic_override is not None else True
        else:
            trace, workload_name, is_synthetic = ExperimentRunner.resolve_workload(config, descriptor)

        if not trace.data:
            raise ValueError(f"Trace data is empty for workload '{workload_name}'.")

        # Determine slot size
        slot_size = (
            config.slot_size
            if config.slot_size is not None
            else max(int(line[3]) for line in trace.data)
        )

        # Determine workload working set (catalog size)
        if config.catalog_size is not None:
            catalog_size = config.catalog_size
        elif "jedi" in workload_name.lower():
            catalog_size = 40000
        elif "snia" in workload_name.lower() or "ibm" in workload_name.lower() or "synthetic" in workload_name.lower() or "synth" in workload_name.lower():
            catalog_size = 10000
        else:
            catalog_size = len(set(line[2] for line in trace.data))

        # Total cache capacity in slots (objects)
        total_slots = max(1, round(catalog_size * config.cache_proportion))

        # Build physical tiers with strict capacity ratio preservation
        if config.custom_tier_configs:
            physical_configs = config.custom_tier_configs
        elif config.num_tiers == 3:
            # 3-tier: DRAM:SSD:NVMe = 1:5:25 (sum = 31 units)
            c1 = max(1, round(total_slots * (1.0 / 31.0)))
            c2 = max(5 * c1, round(total_slots * (5.0 / 31.0)))
            c3 = max(25 * c1, total_slots - c1 - c2)
            dram_size = c1 * slot_size
            ssd_size = c2 * slot_size
            nvme_size = c3 * slot_size
            physical_configs = [
                TierConfig("DRAM", dram_size, 1, 1e-7, 40e9, 20e9),
                TierConfig("SSD", ssd_size, 512, 1e-5, 2e9, 1e9),
                TierConfig("NVME", nvme_size, 512, 1e-4, 500e6, 250e6),
            ]
        else:
            # 2-tier: DRAM:SSD = 1:5 (sum = 6 units)
            c1 = max(1, round(total_slots * (1.0 / 6.0)))
            c2 = max(5 * c1, total_slots - c1)
            dram_size = c1 * slot_size
            ssd_size = c2 * slot_size
            physical_configs = [
                TierConfig("DRAM", dram_size, 1, 1e-7, 40e9, 20e9),
                TierConfig("SSD", ssd_size, 512, 1e-5, 2e9, 1e9),
            ]

        # SimPy environment initialization
        env = simpy.Environment()
        index = Index(env)
        pit = PIT()

        physical_tiers = [
            Tier(
                name=tc.name,
                max_size=tc.max_size,
                granularity=tc.granularity,
                latency=tc.latency,
                read_throughput=tc.read_throughput,
                write_throughput=tc.write_throughput,
                target_occupation=tc.target_occupation,
            )
            for tc in physical_configs
        ]

        # ARC-family policies require the virtual coordinator BOSS tier at index 0
        if descriptor.category == "arc":
            boss = Tier(name="BOSS", max_size=0, granularity=0, latency=0, read_throughput=0, write_throughput=0)
            tiers = [boss] + physical_tiers
            forwarder = Forwarder(env, index, tiers, pit, slot_size, default_tier_index=1)

            # Register coordinator strategy on BOSS
            if config.algorithm_params:
                descriptor.coordinator_class(env, forwarder, boss, **config.algorithm_params)
            else:
                descriptor.coordinator_class(env, forwarder, boss)
            # Register tier policy on each physical storage tier
            for pt in physical_tiers:
                descriptor.tier_policy_class(env, forwarder, pt)

        else:
            # Standard policies and PriorityLRU operate directly on physical tiers
            tiers = physical_tiers
            forwarder = Forwarder(env, index, tiers, pit, slot_size, default_tier_index=0)
            for pt in physical_tiers:
                descriptor.tier_policy_class(env, forwarder, pt)

        # Simulation execution
        log_file = config.output_dir / "logs" / f"{config.algorithm}_{config.seed}.log"
        sim = Simulation([trace], forwarder, env, log_file=str(log_file), logs_enabled=config.logs_enabled)

        t_start = time.time()
        sim.run()
        wall_time = time.time() - t_start

        # Collect standardized metrics
        metrics = SimulationMetrics.from_simulation(
            forwarder=forwarder,
            nb_interests=sim.nb_interests,
            nb_hp=sim.nb_high_priority,
            nb_lp=sim.nb_low_priority,
            wall_time=wall_time,
        )

        result = ExperimentResult.create(
            config=config,
            metrics=metrics,
            workload_name=workload_name,
            is_synthetic=is_synthetic,
        )

        # Save structured results
        result.save_json(config.output_dir / "json")
        result.append_summary_csv(config.output_dir / "experiments_summary.csv")

        return result
