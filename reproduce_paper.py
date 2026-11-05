#!/usr/bin/env python3
"""CLI Entry point to reproduce the experiments from the QM-ARC research paper.

Examples:
    python reproduce_paper.py --requests 1000 --experiments 1 2 3 --workload synthetic
    python reproduce_paper.py --requests 500 --experiments 1 --workload all
"""

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.reproduction import PaperReproductionSuite


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from the QM-ARC research paper (FGCS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=1000,
        help="Number of interest requests per simulation run.",
    )
    parser.add_argument(
        "--workloads",
        "-w",
        nargs="+",
        default=["synthetic"],
        choices=["synthetic", "ibm", "jedi", "all"],
        help="Workloads to evaluate: 'synthetic' (Zipf alpha=1.0), 'ibm' (SNIA fixture), 'jedi' (synthetic video), or 'all'.",
    )
    parser.add_argument(
        "--experiments",
        "-e",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Experiments to run: 1 (capacity sweep), 2 (lambda sweep), 3 (three-tier sweep).",
    )
    parser.add_argument(
        "--seeds",
        "-s",
        nargs="+",
        type=int,
        default=[42],
        help="Random seed(s) for controlled reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results/paper_reproduction",
        help="Base directory to save configs, raw results, aggregations, plots, and comparison reports.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    workloads = ["synthetic", "ibm", "jedi"] if "all" in args.workloads else args.workloads

    print("=" * 70)
    print("QM-ARC Paper Reproduction Suite")
    print("=" * 70)
    print(f"  Requests per Run: {args.requests}")
    print(f"  Workloads:        {', '.join(workloads)}")
    print(f"  Experiments:      {args.experiments}")
    print(f"  Seeds:            {args.seeds}")
    print(f"  Output Directory: {Path(args.output_dir).resolve()}")
    print("=" * 70)

    suite = PaperReproductionSuite(
        base_output_dir=Path(args.output_dir),
        num_requests=args.requests,
        seeds=args.seeds,
    )

    all_results = []
    t_start = time.time()

    for wl in workloads:
        # Experiment 1: Capacity variation (0.5%, 1%, 5%, 10%)
        if 1 in args.experiments:
            res1 = suite.run_experiment_1_capacity_variation(
                workload=wl,
                proportions=[0.005, 0.01, 0.05, 0.10],
            )
            all_results.extend(res1)

        # Experiment 2: Penalty lambda variation (0.2, 0.5, 0.8, 1.0)
        # Note: In the paper, Experiment 2 is centered on Zipf-like workload
        if 2 in args.experiments and (wl == "synthetic" or len(workloads) == 1):
            res2 = suite.run_experiment_2_lambda_variation(
                workload=wl,
                lambdas=[0.20, 0.50, 0.80, 1.00],
                cache_proportion=0.05,
            )
            all_results.extend(res2)

        # Experiment 3: Three-tier architecture (DRAM + SSD + NVMe 1:5:25)
        # Note: In the paper, Experiment 3 is centered on Zipf-like workload
        if 3 in args.experiments and (wl == "synthetic" or len(workloads) == 1):
            res3 = suite.run_experiment_3_three_tier(
                workload=wl,
                proportions=[0.005, 0.01, 0.05, 0.10],
            )
            all_results.extend(res3)

    print("\n" + "=" * 70)
    print("Aggregating Results & Generating Plots...")
    print("=" * 70)

    df = suite.aggregate_results(all_results)
    suite.generate_paper_plots(df)
    comp_df = suite.generate_comparison_table_3(df)

    total_time = time.time() - t_start
    print(f"\nReproduction Complete!")
    print(f"  Total Runs Executed: {len(all_results)}")
    print(f"  Total Elapsed Time:  {total_time:.1f} s")
    print(f"  Raw Metrics:         {suite.raw_dir}")
    print(f"  Aggregated Data:     {suite.agg_dir / 'experiments_aggregated_mean.csv'}")
    print(f"  Plots Generated:     {suite.plots_dir}")
    print(f"  Table 3 Comparison:  {suite.reports_dir / 'table_3_penalty_comparison.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
