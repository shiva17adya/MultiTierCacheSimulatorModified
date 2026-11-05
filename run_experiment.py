#!/usr/bin/env python3
"""CLI Entry point for running reproducible multi-tier cache experiments.

Examples:
    python run_experiment.py --algorithm QM-ARC --workload synthetic --tiers 2 --requests 10000 --seed 42
    python run_experiment.py --algorithm QL-QM-ARC --workload synthetic --tiers 3 --requests 10000 --seed 42
    python run_experiment.py --algorithm QM-ARC --workload synthetic --tiers 2 --requests 5000 --penalty-lambda 0.5
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.config import ExperimentConfig
from experiments.registry import AlgorithmRegistry
from experiments.runner import ExperimentRunner
from experiments.plotting import plot_hit_ratios, plot_penalties, plot_retrieval_latencies


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-tier cache simulation experiment using modular experiment framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="QM-ARC",
        help=f"Cache algorithm to evaluate. Available: {', '.join(AlgorithmRegistry.list_algorithms())}",
    )
    parser.add_argument(
        "--workload",
        "-w",
        type=str,
        default="synthetic",
        help="Workload name ('synthetic', 'jedi', 'snia') or path to custom trace CSV.",
    )
    parser.add_argument(
        "--tiers",
        "-t",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of physical tiers: 2 (DRAM + SSD) or 3 (DRAM + SSD + NVMe).",
    )
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=10000,
        help="Number of requests to simulate.",
    )
    parser.add_argument(
        "--cache-proportion",
        "-c",
        type=float,
        default=0.05,
        help="Cache capacity proportion relative to unique dataset footprint (e.g. 0.005, 0.01, 0.05, 0.10).",
    )
    parser.add_argument(
        "--priority-ratio",
        "-p",
        type=float,
        default=0.20,
        help="Fraction of high-priority requests (for synthetic trace generation).",
    )
    parser.add_argument(
        "--penalty-lambda",
        "--lambda",
        type=float,
        default=0.20,
        help="QoS penalty lambda parameter (ratio of LP penalty to HP penalty, Table 2).",
    )
    parser.add_argument(
        "--catalog-size",
        type=int,
        default=None,
        help="Total unique objects in workload catalog (default: 10000 for synthetic/IBM, 40000 for Jedi).",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Directory to persist JSON metrics, summary CSV, and plots.",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Enable detailed per-packet simulation log output.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Automatically generate evaluation plots from summary CSV after run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Multi-Tier Cache Simulator - Experiment Execution")
    print("=" * 70)
    print(f"  Algorithm:        {args.algorithm}")
    print(f"  Workload:         {args.workload}")
    print(f"  Physical Tiers:   {args.tiers}")
    print(f"  Request Count:    {args.requests}")
    print(f"  Cache Proportion: {args.cache_proportion * 100:.1f}%")
    print(f"  Priority Ratio:   {args.priority_ratio * 100:.1f}% HP")
    print(f"  Penalty Lambda:   {args.penalty_lambda}")
    if args.catalog_size:
        print(f"  Catalog Size:     {args.catalog_size}")
    print(f"  Random Seed:      {args.seed}")
    print(f"  Output Directory: {output_dir.resolve()}")
    print("=" * 70)

    config = ExperimentConfig(
        algorithm=args.algorithm,
        workload=args.workload,
        num_requests=args.requests,
        num_tiers=args.tiers,
        cache_proportion=args.cache_proportion,
        priority_ratio=args.priority_ratio,
        penalty_lambda=args.penalty_lambda,
        catalog_size=args.catalog_size,
        seed=args.seed,
        output_dir=output_dir,
        logs_enabled=args.logs,
    )

    try:
        result = ExperimentRunner.run_experiment(config)
    except Exception as exc:
        print(f"\n[ERROR] Simulation run failed: {exc}")
        sys.exit(1)

    m = result.metrics
    print("\nSimulation Complete!")
    print(f"  Experiment ID:              {result.experiment_id}")
    print(f"  Total Requests:             {m.total_requests} (HP: {m.high_priority_requests}, LP: {m.low_priority_requests})")
    print(f"  Total Hits / Misses:        {m.total_hits} / {m.total_misses}")
    print(f"  Global Hit Ratio:           {m.global_hit_ratio:.4f}")
    print(f"  High-Priority Hit Ratio:    {m.high_priority_hit_ratio:.4f}")
    print(f"  Low-Priority Hit Ratio:     {m.low_priority_hit_ratio:.4f}")
    print(f"  SLA Penalty (HP / LP / Tot): {m.penalty_hp:.1f} / {m.penalty_lp:.1f} / {m.total_penalty:.1f}")
    print(f"  Avg Latency (Global / HP):  {m.avg_retrieval_latency * 1000:.3f} ms / {m.avg_retrieval_latency_hp * 1000:.3f} ms")
    print(f"  Wall Execution Time:        {m.simulation_wall_time_seconds:.2f} s")
    print(f"  JSON Saved:                 {output_dir / 'json' / f'{result.experiment_id}.json'}")
    print(f"  CSV Appended:               {output_dir / 'experiments_summary.csv'}")

    if args.plot:
        csv_file = output_dir / "experiments_summary.csv"
        try:
            plot_hit_ratios(csv_file, output_dir / "plots" / "hit_ratios.png")
            plot_penalties(csv_file, output_dir / "plots" / "penalties.png")
            plot_retrieval_latencies(csv_file, output_dir / "plots" / "latencies.png")
            print(f"  Plots Generated:            {output_dir / 'plots'}")
        except Exception as plot_err:
            print(f"  [WARNING] Plot generation failed: {plot_err}")

    print("=" * 70)


if __name__ == "__main__":
    main()
