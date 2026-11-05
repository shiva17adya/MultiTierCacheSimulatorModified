import csv
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.config import ExperimentConfig, TierConfig
from experiments.registry import AlgorithmRegistry
from experiments.runner import ExperimentRunner
from experiments.results import ExperimentResult


class PaperReproductionSuite:
    """Orchestrates reproduction of Experiments 1, 2, and 3 from the QM-ARC paper."""

    def __init__(
        self,
        base_output_dir: Optional[Path] = None,
        num_requests: int = 1000,
        seeds: Optional[List[int]] = None,
    ):
        self.num_requests = num_requests
        self.seeds = seeds if seeds is not None else [42]
        self.base_dir = Path(base_output_dir) if base_output_dir else Path("results") / "paper_reproduction"
        self.configs_dir = self.base_dir / "configs"
        self.raw_dir = self.base_dir / "raw"
        self.agg_dir = self.base_dir / "aggregated"
        self.plots_dir = self.base_dir / "plots"
        self.reports_dir = self.base_dir / "reports"

        for d in [self.configs_dir, self.raw_dir, self.agg_dir, self.plots_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run_experiment_1_capacity_variation(
        self,
        workload: str = "synthetic",
        proportions: Optional[List[float]] = None,
        algorithms: Optional[List[str]] = None,
    ) -> List[ExperimentResult]:
        """Reproduce Experiment 1: 2-tier DRAM+SSD (1:5 ratio), varying cache capacity (0.5%, 1%, 5%, 10%)."""
        if proportions is None:
            proportions = [0.005, 0.01, 0.05, 0.10]
        if algorithms is None:
            algorithms = ["QM-ARC", "QL-QM-ARC", "M-ARC", "Priority-LRU", "LRU", "LFU", "Random"]

        print(f"\n{'='*70}\n[EXPERIMENT 1] Cache Capacity Variation (2-Tier DRAM+SSD, lambda=0.2, Workload: {workload})\n{'='*70}")
        results = []

        for prop in proportions:
            for seed in self.seeds:
                # Resolve/generate workload ONCE per proportion/seed so all algorithms see exact same trace
                first_algo = algorithms[0]
                sample_cfg = ExperimentConfig(
                    algorithm=first_algo,
                    workload=workload,
                    num_requests=self.num_requests,
                    num_tiers=2,
                    cache_proportion=prop,
                    penalty_lambda=0.20,
                    seed=seed,
                    output_dir=self.raw_dir,
                )
                descriptor = AlgorithmRegistry.get(first_algo)
                trace, workload_name, is_synth = ExperimentRunner.resolve_workload(sample_cfg, descriptor)

                print(f"\n--- Cache Proportion: {prop*100:.1f}%, Seed: {seed}, Workload: {workload_name} (Synthetic: {is_synth}) ---")

                for algo in algorithms:
                    cfg = ExperimentConfig(
                        algorithm=algo,
                        workload=workload,
                        num_requests=self.num_requests,
                        num_tiers=2,
                        cache_proportion=prop,
                        penalty_lambda=0.20,
                        seed=seed,
                        output_dir=self.raw_dir,
                    )
                    # Save configuration metadata
                    cfg_file = self.configs_dir / f"exp1_{workload}_{int(prop*1000)}m_{algo}_s{seed}.json"
                    with open(cfg_file, "w", encoding="utf-8") as f:
                        json.dump(cfg.to_dict(), f, indent=4)

                    # Execute simulation reusing the identical trace
                    res = ExperimentRunner.run_experiment(
                        cfg,
                        existing_trace=trace,
                        workload_name_override=workload_name,
                        is_synthetic_override=is_synth,
                    )
                    results.append(res)
                    m = res.metrics
                    print(f"  [{algo:12s}] GHR: {m.global_hit_ratio:.4f} | HPHR: {m.high_priority_hit_ratio:.4f} | LPHR: {m.low_priority_hit_ratio:.4f} | Pen Tot: {m.total_penalty:.1f} (HP: {m.penalty_hp:.1f}, LP: {m.penalty_lp:.1f})")

        return results

    def run_experiment_2_lambda_variation(
        self,
        workload: str = "synthetic",
        lambdas: Optional[List[float]] = None,
        cache_proportion: float = 0.05,
        algorithms: Optional[List[str]] = None,
    ) -> List[ExperimentResult]:
        """Reproduce Experiment 2: 2-tier DRAM+SSD, fixed 5% capacity, varying lambda (0.2, 0.5, 0.8, 1.0)."""
        if lambdas is None:
            lambdas = [0.20, 0.50, 0.80, 1.00]
        if algorithms is None:
            algorithms = ["QM-ARC", "QL-QM-ARC", "M-ARC", "Priority-LRU", "LRU", "LFU", "Random"]

        print(f"\n{'='*70}\n[EXPERIMENT 2] QoS Penalty Parameter Variation (2-Tier, Capacity: {cache_proportion*100:.1f}%, Workload: {workload})\n{'='*70}")
        results = []

        for lam in lambdas:
            for seed in self.seeds:
                sample_cfg = ExperimentConfig(
                    algorithm=algorithms[0],
                    workload=workload,
                    num_requests=self.num_requests,
                    num_tiers=2,
                    cache_proportion=cache_proportion,
                    penalty_lambda=lam,
                    seed=seed,
                    output_dir=self.raw_dir,
                )
                descriptor = AlgorithmRegistry.get(algorithms[0])
                trace, workload_name, is_synth = ExperimentRunner.resolve_workload(sample_cfg, descriptor)

                print(f"\n--- Penalty Lambda: {lam:.2f}, Seed: {seed}, Workload: {workload_name} ---")

                for algo in algorithms:
                    cfg = ExperimentConfig(
                        algorithm=algo,
                        workload=workload,
                        num_requests=self.num_requests,
                        num_tiers=2,
                        cache_proportion=cache_proportion,
                        penalty_lambda=lam,
                        seed=seed,
                        output_dir=self.raw_dir,
                    )
                    cfg_file = self.configs_dir / f"exp2_{workload}_lam{int(lam*100)}_{algo}_s{seed}.json"
                    with open(cfg_file, "w", encoding="utf-8") as f:
                        json.dump(cfg.to_dict(), f, indent=4)

                    res = ExperimentRunner.run_experiment(
                        cfg,
                        existing_trace=trace,
                        workload_name_override=workload_name,
                        is_synthetic_override=is_synth,
                    )
                    results.append(res)
                    m = res.metrics
                    print(f"  [{algo:12s}] GHR: {m.global_hit_ratio:.4f} | HPHR: {m.high_priority_hit_ratio:.4f} | LPHR: {m.low_priority_hit_ratio:.4f} | Pen Tot: {m.total_penalty:.1f} (HP: {m.penalty_hp:.1f}, LP: {m.penalty_lp:.1f})")

        return results

    def run_experiment_3_three_tier(
        self,
        workload: str = "synthetic",
        proportions: Optional[List[float]] = None,
        algorithms: Optional[List[str]] = None,
    ) -> List[ExperimentResult]:
        """Reproduce Experiment 3: 3-tier DRAM+SSD+NVMe (1:5:25 ratio), varying cache capacity (0.5%, 1%, 5%, 10%)."""
        if proportions is None:
            proportions = [0.005, 0.01, 0.05, 0.10]
        # In the paper, Section 4.2.3, Priority LRU is excluded because it is not mapped to three tiers
        if algorithms is None:
            algorithms = ["QM-ARC", "QL-QM-ARC", "M-ARC", "LRU", "LFU", "Random"]

        print(f"\n{'='*70}\n[EXPERIMENT 3] Three-Tier Architecture (DRAM+SSD+NVMe 1:5:25, lambda=0.2, Workload: {workload})\n{'='*70}")
        results = []

        for prop in proportions:
            for seed in self.seeds:
                sample_cfg = ExperimentConfig(
                    algorithm=algorithms[0],
                    workload=workload,
                    num_requests=self.num_requests,
                    num_tiers=3,
                    cache_proportion=prop,
                    penalty_lambda=0.20,
                    seed=seed,
                    output_dir=self.raw_dir,
                )
                descriptor = AlgorithmRegistry.get(algorithms[0])
                trace, workload_name, is_synth = ExperimentRunner.resolve_workload(sample_cfg, descriptor)

                print(f"\n--- 3-Tier Cache Proportion: {prop*100:.1f}%, Seed: {seed}, Workload: {workload_name} ---")

                for algo in algorithms:
                    cfg = ExperimentConfig(
                        algorithm=algo,
                        workload=workload,
                        num_requests=self.num_requests,
                        num_tiers=3,
                        cache_proportion=prop,
                        penalty_lambda=0.20,
                        seed=seed,
                        output_dir=self.raw_dir,
                    )
                    cfg_file = self.configs_dir / f"exp3_{workload}_{int(prop*1000)}m_{algo}_s{seed}.json"
                    with open(cfg_file, "w", encoding="utf-8") as f:
                        json.dump(cfg.to_dict(), f, indent=4)

                    res = ExperimentRunner.run_experiment(
                        cfg,
                        existing_trace=trace,
                        workload_name_override=workload_name,
                        is_synthetic_override=is_synth,
                    )
                    results.append(res)
                    m = res.metrics
                    print(f"  [{algo:12s}] GHR: {m.global_hit_ratio:.4f} | HPHR: {m.high_priority_hit_ratio:.4f} | LPHR: {m.low_priority_hit_ratio:.4f} | Pen Tot: {m.total_penalty:.1f} (HP: {m.penalty_hp:.1f}, LP: {m.penalty_lp:.1f})")

        return results

    def aggregate_results(self, all_results: List[ExperimentResult]) -> pd.DataFrame:
        """Aggregate raw results into a summary DataFrame with mean and std per configuration."""
        rows = []
        for r in all_results:
            rows.append({
                "experiment_id": r.experiment_id,
                "algorithm": r.algorithm,
                "workload": r.workload_name,
                "is_synthetic": r.is_synthetic,
                "num_tiers": r.config.num_tiers,
                "cache_proportion": r.config.cache_proportion,
                "penalty_lambda": r.config.penalty_lambda,
                "seed": r.config.seed,
                "requests": r.metrics.total_requests,
                "hp_requests": r.metrics.high_priority_requests,
                "lp_requests": r.metrics.low_priority_requests,
                "hits": r.metrics.total_hits,
                "misses": r.metrics.total_misses,
                "ghr": r.metrics.global_hit_ratio,
                "hphr": r.metrics.high_priority_hit_ratio,
                "lphr": r.metrics.low_priority_hit_ratio,
                "penalty_hp": r.metrics.penalty_hp,
                "penalty_lp": r.metrics.penalty_lp,
                "total_penalty": r.metrics.total_penalty,
                "avg_latency_ms": r.metrics.avg_retrieval_latency * 1000.0,
                "avg_latency_hp_ms": r.metrics.avg_retrieval_latency_hp * 1000.0,
                "avg_latency_lp_ms": r.metrics.avg_retrieval_latency_lp * 1000.0,
                "wall_time_s": r.metrics.simulation_wall_time_seconds,
            })
        df = pd.DataFrame(rows)
        raw_csv = self.agg_dir / "all_experiments_raw.csv"
        df.to_csv(raw_csv, index=False)

        # Aggregated mean & std grouped by (algorithm, workload, requests, num_tiers, cache_proportion, penalty_lambda)
        group_cols = ["algorithm", "workload", "requests", "num_tiers", "cache_proportion", "penalty_lambda"]
        metric_cols = ["ghr", "hphr", "lphr", "penalty_hp", "penalty_lp", "total_penalty", "avg_latency_ms"]
        agg_mean = df.groupby(group_cols)[metric_cols].mean().reset_index()
        agg_csv = self.agg_dir / "experiments_aggregated_mean.csv"
        agg_mean.to_csv(agg_csv, index=False)
        return df

    def generate_paper_plots(self, df: pd.DataFrame):
        """Generate high-resolution plots mirroring Figures 2, 6, and 7 of the paper."""
        plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})
        colors = {
            "QM-ARC": "#1f77b4",
            "QL-QM-ARC": "#2ca02c",
            "M-ARC": "#ff7f0e",
            "Priority-LRU": "#d62728",
            "LRU": "#9467bd",
            "LFU": "#8c564b",
            "Random": "#7f7f7f",
        }

        # -------------------------------------------------------------
        # Plot Exp 1 (2-tier, lambda=0.2): Hit Rates vs Cache Proportion
        # -------------------------------------------------------------
        exp1_df = df[(df["num_tiers"] == 2) & (df["penalty_lambda"] == 0.20)]
        if not exp1_df.empty:
            for wl in exp1_df["workload"].unique():
                sub = exp1_df[exp1_df["workload"] == wl]

                # (a) Global Hit Rate
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
                for algo in sorted(sub["algorithm"].unique()):
                    a_df = (
                        sub[sub["algorithm"] == algo]
                        .groupby("cache_proportion")[["ghr", "hphr", "lphr", "total_penalty", "penalty_hp", "penalty_lp"]]
                        .mean()
                        .reset_index()
                        .sort_values("cache_proportion")
                    )
                    c = colors.get(algo, "black")
                    p_labels = [f"{p*100:.1f}%" for p in a_df["cache_proportion"]]
                    ax1.plot(p_labels, a_df["ghr"], marker="o", label=algo, color=c, linewidth=2)
                    ax2.plot(p_labels, a_df["hphr"], marker="s", label=algo, color=c, linewidth=2)
                    ax3.plot(p_labels, a_df["lphr"], marker="^", label=algo, color=c, linewidth=2)

                ax1.set_title("Global Cache Hit Rate", fontweight="bold")
                ax1.set_xlabel("Cache Proportion")
                ax1.set_ylabel("Hit Rate (0.0 - 1.0)")
                ax1.set_ylim(0, 1.05)
                ax1.grid(True, linestyle="--", alpha=0.5)

                ax2.set_title("High-Priority Hit Rate", fontweight="bold")
                ax2.set_xlabel("Cache Proportion")
                ax2.set_ylabel("Hit Rate (0.0 - 1.0)")
                ax2.set_ylim(0, 1.05)
                ax2.grid(True, linestyle="--", alpha=0.5)

                ax3.set_title("Low-Priority Hit Rate", fontweight="bold")
                ax3.set_xlabel("Cache Proportion")
                ax3.set_ylabel("Hit Rate (0.0 - 1.0)")
                ax3.set_ylim(0, 1.05)
                ax3.grid(True, linestyle="--", alpha=0.5)
                ax3.legend(loc="best", fontsize=9)

                plt.suptitle(f"Exp 1: Hit Rates vs Cache Proportion ({wl})", fontsize=14, fontweight="bold")
                plt.tight_layout()
                fig_path = self.plots_dir / f"exp1_{wl}_hit_rates.png"
                plt.savefig(fig_path, dpi=300)
                plt.close(fig)

                # (b) Penalties
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
                for algo in sorted(sub["algorithm"].unique()):
                    a_df = (
                        sub[sub["algorithm"] == algo]
                        .groupby("cache_proportion")[["ghr", "hphr", "lphr", "total_penalty", "penalty_hp", "penalty_lp"]]
                        .mean()
                        .reset_index()
                        .sort_values("cache_proportion")
                    )
                    c = colors.get(algo, "black")
                    p_labels = [f"{p*100:.1f}%" for p in a_df["cache_proportion"]]
                    ax1.plot(p_labels, a_df["total_penalty"], marker="o", label=algo, color=c, linewidth=2)
                    ax2.plot(p_labels, a_df["penalty_hp"], marker="s", label=algo, color=c, linewidth=2)
                    ax3.plot(p_labels, a_df["penalty_lp"], marker="^", label=algo, color=c, linewidth=2)

                ax1.set_title("Total SLA Penalty", fontweight="bold")
                ax1.set_xlabel("Cache Proportion")
                ax1.set_ylabel("Penalty")
                ax1.grid(True, linestyle="--", alpha=0.5)

                ax2.set_title("High-Priority Penalty", fontweight="bold")
                ax2.set_xlabel("Cache Proportion")
                ax2.set_ylabel("Penalty")
                ax2.grid(True, linestyle="--", alpha=0.5)

                ax3.set_title("Low-Priority Penalty", fontweight="bold")
                ax3.set_xlabel("Cache Proportion")
                ax3.set_ylabel("Penalty")
                ax3.grid(True, linestyle="--", alpha=0.5)
                ax3.legend(loc="best", fontsize=9)

                plt.suptitle(f"Exp 1: SLA Penalties vs Cache Proportion ({wl})", fontsize=14, fontweight="bold")
                plt.tight_layout()
                fig_pen_path = self.plots_dir / f"exp1_{wl}_penalties.png"
                plt.savefig(fig_pen_path, dpi=300)
                plt.close(fig)

        # -------------------------------------------------------------
        # Plot Exp 2: Varying Lambda (Figure 6)
        # -------------------------------------------------------------
        exp2_df = df[(df["num_tiers"] == 2) & (df["cache_proportion"] == 0.05)]
        if len(exp2_df["penalty_lambda"].unique()) > 1:
            for wl in exp2_df["workload"].unique():
                sub = exp2_df[exp2_df["workload"] == wl]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                for algo in sorted(sub["algorithm"].unique()):
                    a_df = (
                        sub[sub["algorithm"] == algo]
                        .groupby("penalty_lambda")[["total_penalty", "penalty_hp"]]
                        .mean()
                        .reset_index()
                        .sort_values("penalty_lambda")
                    )
                    c = colors.get(algo, "black")
                    l_labels = [f"{l:.1f}" for l in a_df["penalty_lambda"]]
                    ax1.plot(l_labels, a_df["total_penalty"], marker="o", label=algo, color=c, linewidth=2)
                    ax2.plot(l_labels, a_df["penalty_hp"], marker="s", label=algo, color=c, linewidth=2)

                ax1.set_title("Total Penalty vs Lambda", fontweight="bold")
                ax1.set_xlabel("Lambda (LP / HP Penalty Ratio)")
                ax1.set_ylabel("Total Penalty")
                ax1.grid(True, linestyle="--", alpha=0.5)
                ax1.legend(loc="best", fontsize=9)

                ax2.set_title("High-Priority Penalty vs Lambda", fontweight="bold")
                ax2.set_xlabel("Lambda (LP / HP Penalty Ratio)")
                ax2.set_ylabel("HP Penalty")
                ax2.grid(True, linestyle="--", alpha=0.5)

                plt.suptitle(f"Exp 2: Penalty vs Lambda ({wl}, 5% Cache)", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"exp2_{wl}_lambda_sweep.png", dpi=300)
                plt.close(fig)

        # -------------------------------------------------------------
        # Plot Exp 3: Three-Tier System (Figure 7)
        # -------------------------------------------------------------
        exp3_df = df[df["num_tiers"] == 3]
        if not exp3_df.empty:
            for wl in exp3_df["workload"].unique():
                sub = exp3_df[exp3_df["workload"] == wl]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                for algo in sorted(sub["algorithm"].unique()):
                    a_df = (
                        sub[sub["algorithm"] == algo]
                        .groupby("cache_proportion")[["ghr", "total_penalty"]]
                        .mean()
                        .reset_index()
                        .sort_values("cache_proportion")
                    )
                    c = colors.get(algo, "black")
                    p_labels = [f"{p*100:.1f}%" for p in a_df["cache_proportion"]]
                    ax1.plot(p_labels, a_df["ghr"], marker="o", label=algo, color=c, linewidth=2)
                    ax2.plot(p_labels, a_df["total_penalty"], marker="s", label=algo, color=c, linewidth=2)

                ax1.set_title("3-Tier Global Hit Rate", fontweight="bold")
                ax1.set_xlabel("Cache Proportion")
                ax1.set_ylabel("Hit Rate (0.0 - 1.0)")
                ax1.set_ylim(0, 1.05)
                ax1.grid(True, linestyle="--", alpha=0.5)
                ax1.legend(loc="best", fontsize=9)

                ax2.set_title("3-Tier Total Penalty", fontweight="bold")
                ax2.set_xlabel("Cache Proportion")
                ax2.set_ylabel("Total Penalty")
                ax2.grid(True, linestyle="--", alpha=0.5)

                plt.suptitle(f"Exp 3: Three-Tier Architecture DRAM+SSD+NVMe ({wl})", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"exp3_{wl}_three_tier.png", dpi=300)
                plt.close(fig)

    def generate_comparison_table_3(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Table 3: Percentage differences in penalties compared to QM-ARC.

        Formula: Difference(%) = ((QM-ARC_Penalty - Strategy_Penalty) / QM-ARC_Penalty) * 100
        Negative value indicates QM-ARC had a lower penalty (QM-ARC performed better).
        Positive value indicates Strategy had a lower penalty (QM-ARC performed worse).
        """
        table_rows = []
        exp1_df = df[(df["num_tiers"] == 2) & (df["penalty_lambda"] == 0.20)]

        for prop in sorted(exp1_df["cache_proportion"].unique()):
            for wl in exp1_df["workload"].unique():
                sub = exp1_df[(exp1_df["cache_proportion"] == prop) & (exp1_df["workload"] == wl)]
                qm_rows = sub[sub["algorithm"] == "QM-ARC"]
                if qm_rows.empty:
                    continue
                qm_pen_hp = qm_rows["penalty_hp"].mean()
                qm_pen_lp = qm_rows["penalty_lp"].mean()

                for algo in sorted(sub["algorithm"].unique()):
                    if algo == "QM-ARC":
                        continue
                    a_rows = sub[sub["algorithm"] == algo]
                    a_pen_hp = a_rows["penalty_hp"].mean()
                    a_pen_lp = a_rows["penalty_lp"].mean()

                    # Percentage difference relative to Strategy (Paper Table 3 formula: (QM - Strat) / Strat * 100)
                    # Negative value indicates QM-ARC had a lower penalty (QM-ARC performed better).
                    diff_hp = ((qm_pen_hp - a_pen_hp) / a_pen_hp * 100.0) if a_pen_hp > 0 else 0.0
                    diff_lp = ((qm_pen_lp - a_pen_lp) / a_pen_lp * 100.0) if a_pen_lp > 0 else 0.0

                    table_rows.append({
                        "cache_proportion": f"{prop*100:.1f}%",
                        "workload": wl,
                        "strategy": algo,
                        "hp_penalty_diff_pct": round(diff_hp, 2),
                        "lp_penalty_diff_pct": round(diff_lp, 2),
                        "qm_hp_penalty": qm_pen_hp,
                        "strategy_hp_penalty": a_pen_hp,
                        "qm_lp_penalty": qm_pen_lp,
                        "strategy_lp_penalty": a_pen_lp,
                    })

        res_df = pd.DataFrame(table_rows)
        res_df.to_csv(self.reports_dir / "table_3_penalty_comparison.csv", index=False)
        return res_df
