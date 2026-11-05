import csv
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from experiments.config import ExperimentConfig
from experiments.metrics import SimulationMetrics


@dataclass
class ExperimentResult:
    """Encapsulates the complete result of an evaluation run with persistence capabilities."""
    experiment_id: str
    timestamp: str
    algorithm: str
    workload_name: str
    is_synthetic: bool
    config: ExperimentConfig
    metrics: SimulationMetrics

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        metrics: SimulationMetrics,
        workload_name: str,
        is_synthetic: bool,
    ) -> "ExperimentResult":
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        algo_slug = config.algorithm.replace("-", "_").lower()
        exp_id = f"exp_{algo_slug}_{config.num_tiers}tier_{ts_str}_{uid}"
        return cls(
            experiment_id=exp_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            algorithm=config.algorithm,
            workload_name=workload_name,
            is_synthetic=is_synthetic,
            config=config,
            metrics=metrics,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "algorithm": self.algorithm,
            "workload_name": self.workload_name,
            "is_synthetic": self.is_synthetic,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    def save_json(self, destination_dir: Optional[Path] = None) -> Path:
        """Write detailed experiment metadata and metrics to a structured JSON file."""
        target_dir = Path(destination_dir) if destination_dir else self.config.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{self.experiment_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
        return file_path

    def append_summary_csv(self, csv_file_path: Optional[Path] = None) -> Path:
        """Append key scalar metrics to a tabular summary CSV."""
        target_file = (
            Path(csv_file_path)
            if csv_file_path
            else (self.config.output_dir / "experiments_summary.csv")
        )
        target_file.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "experiment_id",
            "timestamp",
            "algorithm",
            "workload_name",
            "is_synthetic",
            "num_tiers",
            "total_requests",
            "high_priority_requests",
            "low_priority_requests",
            "total_hits",
            "total_misses",
            "global_hit_ratio",
            "high_priority_hit_ratio",
            "low_priority_hit_ratio",
            "penalty_hp",
            "penalty_lp",
            "total_penalty",
            "avg_retrieval_latency",
            "avg_retrieval_latency_hp",
            "avg_retrieval_latency_lp",
            "wall_time_seconds",
        ]

        file_exists = target_file.is_file()
        with open(target_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "experiment_id": self.experiment_id,
                "timestamp": self.timestamp,
                "algorithm": self.algorithm,
                "workload_name": self.workload_name,
                "is_synthetic": self.is_synthetic,
                "num_tiers": self.config.num_tiers,
                "total_requests": self.metrics.total_requests,
                "high_priority_requests": self.metrics.high_priority_requests,
                "low_priority_requests": self.metrics.low_priority_requests,
                "total_hits": self.metrics.total_hits,
                "total_misses": self.metrics.total_misses,
                "global_hit_ratio": round(self.metrics.global_hit_ratio, 6),
                "high_priority_hit_ratio": round(self.metrics.high_priority_hit_ratio, 6),
                "low_priority_hit_ratio": round(self.metrics.low_priority_hit_ratio, 6),
                "penalty_hp": self.metrics.penalty_hp,
                "penalty_lp": self.metrics.penalty_lp,
                "total_penalty": self.metrics.total_penalty,
                "avg_retrieval_latency": round(self.metrics.avg_retrieval_latency, 8),
                "avg_retrieval_latency_hp": round(self.metrics.avg_retrieval_latency_hp, 8),
                "avg_retrieval_latency_lp": round(self.metrics.avg_retrieval_latency_lp, 8),
                "wall_time_seconds": round(self.metrics.simulation_wall_time_seconds, 3),
            })
        return target_file
