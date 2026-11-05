from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class TierConfig:
    """Hardware and capacity configuration for a single physical storage tier."""
    name: str
    max_size: int  # in bytes
    granularity: int = 512  # in bytes
    latency: float = 1e-5  # in seconds
    read_throughput: float = 2e9  # in bytes/sec
    write_throughput: float = 1e9  # in bytes/sec
    target_occupation: float = 1.0


@dataclass
class ExperimentConfig:
    """Comprehensive experiment parameters defining an evaluation run."""
    algorithm: str  # e.g. "QM-ARC", "M-ARC", "QL-QM-ARC", "LRU", "LFU", "Random", "PriorityLRU"
    workload: str = "synthetic"  # "synthetic", "jedi", "snia", or path to custom CSV
    num_requests: int = 10000
    num_tiers: int = 2  # 2 or 3
    cache_proportion: float = 0.05  # fraction of unique catalog footprint (e.g. 0.005, 0.01, 0.05, 0.10)
    priority_ratio: float = 0.20  # fraction of high-priority requests
    penalty_lambda: float = 0.20  # QoS penalty lambda parameter (Table 2: 0.2 for exp 1 & 3; 0.2-1.0 for exp 2)
    catalog_size: Optional[int] = None  # Total unique objects in workload catalog (e.g. 10000 for Zipf, 40000 for Jedi)
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results"))
    logs_enabled: bool = False
    custom_tier_configs: Optional[List[TierConfig]] = None
    slot_size: Optional[int] = None  # detected automatically from trace if None
    algorithm_params: Optional[Dict[str, Any]] = None

    @classmethod
    def create_two_tier(
        cls,
        algorithm: str,
        total_cache_size: int,
        workload: str = "synthetic",
        num_requests: int = 10000,
        penalty_lambda: float = 0.20,
        seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> "ExperimentConfig":
        """Factory for paper-standard 2-tier DRAM + Optane SSD setup (DRAM:SSD = 1:5)."""
        dram_size = int(total_cache_size * (1.0 / 6.0))
        ssd_size = total_cache_size - dram_size

        tiers = [
            TierConfig(name="DRAM", max_size=dram_size, granularity=1, latency=1e-7,
                       read_throughput=40e9, write_throughput=20e9),
            TierConfig(name="SSD", max_size=ssd_size, granularity=512, latency=1e-5,
                       read_throughput=2e9, write_throughput=1e9),
        ]
        return cls(
            algorithm=algorithm,
            workload=workload,
            num_requests=num_requests,
            num_tiers=2,
            penalty_lambda=penalty_lambda,
            seed=seed,
            output_dir=output_dir if output_dir is not None else Path("results"),
            custom_tier_configs=tiers,
        )

    @classmethod
    def create_three_tier(
        cls,
        algorithm: str,
        total_cache_size: int,
        workload: str = "synthetic",
        num_requests: int = 10000,
        penalty_lambda: float = 0.20,
        seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> "ExperimentConfig":
        """Factory for paper-standard 3-tier DRAM + SSD + NVMe setup (DRAM:SSD:NVMe = 1:5:25)."""
        unit = total_cache_size / 31.0
        dram_size = int(round(unit * 1.0))
        ssd_size = int(round(unit * 5.0))
        nvme_size = total_cache_size - dram_size - ssd_size

        tiers = [
            TierConfig(name="DRAM", max_size=dram_size, granularity=1, latency=1e-7,
                       read_throughput=40e9, write_throughput=20e9),
            TierConfig(name="SSD", max_size=ssd_size, granularity=512, latency=1e-5,
                       read_throughput=2e9, write_throughput=1e9),
            TierConfig(name="NVME", max_size=nvme_size, granularity=512, latency=1e-4,
                       read_throughput=500e6, write_throughput=250e6),
        ]
        return cls(
            algorithm=algorithm,
            workload=workload,
            num_requests=num_requests,
            num_tiers=3,
            penalty_lambda=penalty_lambda,
            seed=seed,
            output_dir=output_dir if output_dir is not None else Path("results"),
            custom_tier_configs=tiers,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "workload": self.workload,
            "num_requests": self.num_requests,
            "num_tiers": self.num_tiers,
            "cache_proportion": self.cache_proportion,
            "priority_ratio": self.priority_ratio,
            "penalty_lambda": self.penalty_lambda,
            "catalog_size": self.catalog_size,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "logs_enabled": self.logs_enabled,
            "slot_size": self.slot_size,
            "algorithm_params": self.algorithm_params,
            "tier_configs": [
                {
                    "name": t.name,
                    "max_size": t.max_size,
                    "latency": t.latency,
                    "read_throughput": t.read_throughput,
                    "write_throughput": t.write_throughput,
                }
                for t in (self.custom_tier_configs or [])
            ],
        }
