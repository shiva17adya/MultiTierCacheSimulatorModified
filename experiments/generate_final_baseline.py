"""Generate the complete, verified final baseline for QM-ARC paper reproduction.

Runs all 7 algorithms across 2-tier, 3-tier, lambda variations, and multi-seed statistical trials.
Outputs to results/final_baseline/ with strict configuration isolation.
"""

import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Any, Optional

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.config import ExperimentConfig, TierConfig
from experiments.registry import AlgorithmRegistry
from experiments.runner import ExperimentRunner
from experiments.results import ExperimentResult
from common.penalty import set_penalty_lambda


class FinalBaselineSuite:
    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            # Default to repo root / ../results/final_baseline
            self.base_dir = Path(__file__).resolve().parents[1] / "results" / "final_baseline"
        else:
            self.base_dir = Path(output_dir)

        self.raw_dir = self.base_dir / "raw"
        self.raw_json_dir = self.raw_dir / "json"
        self.configs_dir = self.base_dir / "configs"
        self.agg_dir = self.base_dir / "aggregated"
        self.plots_dir = self.base_dir / "plots"
        self.reports_dir = self.base_dir / "reports"
        self.stat_dir = self.base_dir / "statistical"

        for d in [self.raw_dir, self.raw_json_dir, self.configs_dir, self.agg_dir, self.plots_dir, self.reports_dir, self.stat_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run_principal_experiments(self) -> List[ExperimentResult]:
        """Execute Experiments 1, 2, and 3 across authentic Zipf, IBM fixture, and Jedi approx."""
        algorithms_all = ["QM-ARC", "QL-QM-ARC", "M-ARC", "Priority-LRU", "LRU", "LFU", "Random"]
        algorithms_3tier = ["QM-ARC", "QL-QM-ARC", "M-ARC", "LRU", "LFU", "Random"]  # Priority-LRU excluded per Sec 4.2.3
        proportions = [0.005, 0.01, 0.05, 0.10]
        lambdas = [0.20, 0.50, 0.80, 1.00]

        all_results: List[ExperimentResult] = []

        # -------------------------------------------------------------
        # 1. Experiment 1: Two-Tier Capacity Variation
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[FINAL BASELINE] Running Experiment 1: Two-Tier Capacity Variation")
        print("=" * 70)

        workloads = [
            ("synthetic", 1000, 42),
            ("ibm", 500, 42),
            ("jedi", 500, 42),
        ]

        for wl_key, n_reqs, seed in workloads:
            for prop in proportions:
                # Generate shared trace
                sample_cfg = ExperimentConfig(
                    algorithm=algorithms_all[0],
                    workload=wl_key,
                    num_requests=n_reqs,
                    num_tiers=2,
                    cache_proportion=prop,
                    priority_ratio=0.20,
                    penalty_lambda=0.20,
                    seed=seed,
                    output_dir=self.raw_dir,
                )
                base_desc = AlgorithmRegistry.get(algorithms_all[0])
                shared_trace, wl_name, is_synth = ExperimentRunner.resolve_workload(sample_cfg, base_desc)

                for algo in algorithms_all:
                    cfg = ExperimentConfig(
                        algorithm=algo,
                        workload=wl_key,
                        num_requests=n_reqs,
                        num_tiers=2,
                        cache_proportion=prop,
                        priority_ratio=0.20,
                        penalty_lambda=0.20,
                        seed=seed,
                        output_dir=self.raw_dir,
                    )
                    cfg_file = self.configs_dir / f"exp1_{wl_key}_{int(prop*1000)}m_{algo}_s{seed}.json"
                    with open(cfg_file, "w", encoding="utf-8") as f:
                        json.dump(cfg.to_dict(), f, indent=4)

                    res = ExperimentRunner.run_experiment(
                        cfg,
                        existing_trace=shared_trace,
                        workload_name_override=wl_name,
                        is_synthetic_override=is_synth,
                    )
                    json_path = self.raw_json_dir / f"exp1_{wl_key}_{int(prop*1000)}m_{algo}_s{seed}.json"
                    res.save_json(json_path)
                    all_results.append(res)
                    m = res.metrics
                    print(f"  [Exp1 {wl_key} {prop*100:4.1f}% | {algo:12s}] GHR: {m.global_hit_ratio:.4f} | Pen Tot: {m.total_penalty:.1f} (HP: {m.penalty_hp:.1f}, LP: {m.penalty_lp:.1f})")

        # -------------------------------------------------------------
        # 2. Experiment 2: Lambda Variation (Synthetic Zipf, 5% cache)
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[FINAL BASELINE] Running Experiment 2: Lambda Sweep (5% Cache)")
        print("=" * 70)

        sample_cfg = ExperimentConfig(
            algorithm=algorithms_all[0],
            workload="synthetic",
            num_requests=1000,
            num_tiers=2,
            cache_proportion=0.05,
            priority_ratio=0.20,
            penalty_lambda=0.20,
            seed=42,
            output_dir=self.raw_dir,
        )
        base_desc = AlgorithmRegistry.get(algorithms_all[0])
        shared_trace, wl_name, is_synth = ExperimentRunner.resolve_workload(sample_cfg, base_desc)

        for lam in lambdas:
            for algo in algorithms_all:
                cfg = ExperimentConfig(
                    algorithm=algo,
                    workload="synthetic",
                    num_requests=1000,
                    num_tiers=2,
                    cache_proportion=0.05,
                    priority_ratio=0.20,
                    penalty_lambda=lam,
                    seed=42,
                    output_dir=self.raw_dir,
                )
                cfg_file = self.configs_dir / f"exp2_synthetic_lam{int(lam*100)}_{algo}_s42.json"
                with open(cfg_file, "w", encoding="utf-8") as f:
                    json.dump(cfg.to_dict(), f, indent=4)

                res = ExperimentRunner.run_experiment(
                    cfg,
                    existing_trace=shared_trace,
                    workload_name_override=wl_name,
                    is_synthetic_override=is_synth,
                )
                json_path = self.raw_json_dir / f"exp2_synthetic_lam{int(lam*100)}_{algo}_s42.json"
                res.save_json(json_path)
                all_results.append(res)
                m = res.metrics
                print(f"  [Exp2 lam={lam:3.1f} | {algo:12s}] GHR: {m.global_hit_ratio:.4f} | Pen Tot: {m.total_penalty:.1f} (HP: {m.penalty_hp:.1f}, LP: {m.penalty_lp:.1f})")

        # -------------------------------------------------------------
        # 3. Experiment 3: Three-Tier Hierarchy (DRAM:SSD:NVMe = 1:5:25)
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[FINAL BASELINE] Running Experiment 3: Three-Tier Hierarchy (1:5:25)")
        print("=" * 70)

        for prop in proportions:
            sample_cfg = ExperimentConfig(
                algorithm=algorithms_3tier[0],
                workload="synthetic",
                num_requests=1000,
                num_tiers=3,
                cache_proportion=prop,
                priority_ratio=0.20,
                penalty_lambda=0.20,
                seed=42,
                output_dir=self.raw_dir,
            )
            base_desc = AlgorithmRegistry.get(algorithms_3tier[0])
            shared_trace, wl_name, is_synth = ExperimentRunner.resolve_workload(sample_cfg, base_desc)

            for algo in algorithms_3tier:
                cfg = ExperimentConfig(
                    algorithm=algo,
                    workload="synthetic",
                    num_requests=1000,
                    num_tiers=3,
                    cache_proportion=prop,
                    priority_ratio=0.20,
                    penalty_lambda=0.20,
                    seed=42,
                    output_dir=self.raw_dir,
                )
                cfg_file = self.configs_dir / f"exp3_synthetic_{int(prop*1000)}m_{algo}_s42.json"
                with open(cfg_file, "w", encoding="utf-8") as f:
                    json.dump(cfg.to_dict(), f, indent=4)

                res = ExperimentRunner.run_experiment(
                    cfg,
                    existing_trace=shared_trace,
                    workload_name_override=wl_name,
                    is_synthetic_override=is_synth,
                )
                json_path = self.raw_json_dir / f"exp3_synthetic_{int(prop*1000)}m_{algo}_s42.json"
                res.save_json(json_path)
                all_results.append(res)
                m = res.metrics
                print(f"  [Exp3 3-tier {prop*100:4.1f}% | {algo:12s}] GHR: {m.global_hit_ratio:.4f} | Pen Tot: {m.total_penalty:.1f} (HP: {m.penalty_hp:.1f}, LP: {m.penalty_lp:.1f})")

        return all_results

    def run_multi_seed_statistical_suite(self, seeds: Optional[List[int]] = None) -> pd.DataFrame:
        """Run independent multi-seed statistical trials with genuine independent traces."""
        if seeds is None:
            seeds = [42, 101, 202, 303, 404]

        algorithms = ["QM-ARC", "QL-QM-ARC", "M-ARC", "Priority-LRU", "LRU", "LFU", "Random"]
        proportions = [0.005, 0.05]
        num_requests = 1000

        print("\n" + "=" * 70)
        print(f"[FINAL BASELINE] Running Multi-Seed Statistical Suite ({len(seeds)} independent seeds, N={num_requests})")
        print("=" * 70)

        rows = []
        for prop in proportions:
            for seed in seeds:
                sample_cfg = ExperimentConfig(
                    algorithm=algorithms[0],
                    workload="synthetic",
                    num_requests=num_requests,
                    num_tiers=2,
                    cache_proportion=prop,
                    priority_ratio=0.20,
                    penalty_lambda=0.20,
                    seed=seed,
                    output_dir=self.raw_dir,
                )
                base_desc = AlgorithmRegistry.get(algorithms[0])
                shared_trace, wl_name, is_synth = ExperimentRunner.resolve_workload(sample_cfg, base_desc)

                for algo in algorithms:
                    cfg = ExperimentConfig(
                        algorithm=algo,
                        workload="synthetic",
                        num_requests=num_requests,
                        num_tiers=2,
                        cache_proportion=prop,
                        priority_ratio=0.20,
                        penalty_lambda=0.20,
                        seed=seed,
                        output_dir=self.raw_dir,
                    )
                    res = ExperimentRunner.run_experiment(
                        cfg,
                        existing_trace=shared_trace,
                        workload_name_override=wl_name,
                        is_synthetic_override=is_synth,
                    )
                    m = res.metrics
                    rows.append({
                        "seed": seed,
                        "cache_proportion": prop,
                        "algorithm": algo,
                        "requests": num_requests,
                        "ghr": m.global_hit_ratio,
                        "hphr": m.high_priority_hit_ratio,
                        "lphr": m.low_priority_hit_ratio,
                        "penalty_hp": m.penalty_hp,
                        "penalty_lp": m.penalty_lp,
                        "total_penalty": m.total_penalty,
                        "avg_latency": m.avg_retrieval_latency,
                    })

        stat_raw_df = pd.DataFrame(rows)
        stat_raw_df.to_csv(self.stat_dir / "statistical_trials_raw.csv", index=False)

        # Statistical analysis: Mean, Std Dev, 95% Confidence Interval (t-distribution df=4, t_crit=2.776445)
        t_crit = 2.776445
        stat_summary = []
        for (prop, algo), grp in stat_raw_df.groupby(["cache_proportion", "algorithm"]):
            n = len(grp)
            m_ghr = grp["ghr"].mean()
            s_ghr = grp["ghr"].std(ddof=1)
            m_hp = grp["penalty_hp"].mean()
            s_hp = grp["penalty_hp"].std(ddof=1)
            ci_hp = t_crit * (s_hp / math.sqrt(n)) if n > 1 else 0.0

            m_lp = grp["penalty_lp"].mean()
            s_lp = grp["penalty_lp"].std(ddof=1)
            ci_lp = t_crit * (s_lp / math.sqrt(n)) if n > 1 else 0.0

            m_tot = grp["total_penalty"].mean()
            s_tot = grp["total_penalty"].std(ddof=1)
            ci_tot = t_crit * (s_tot / math.sqrt(n)) if n > 1 else 0.0

            stat_summary.append({
                "cache_proportion": f"{prop*100:.1f}%",
                "algorithm": algo,
                "trials": n,
                "mean_ghr": round(m_ghr, 4),
                "std_ghr": round(s_ghr, 4),
                "mean_penalty_hp": round(m_hp, 2),
                "std_penalty_hp": round(s_hp, 2),
                "ci95_penalty_hp": round(ci_hp, 2),
                "mean_penalty_lp": round(m_lp, 2),
                "std_penalty_lp": round(s_lp, 2),
                "ci95_penalty_lp": round(ci_lp, 2),
                "mean_total_penalty": round(m_tot, 2),
                "std_total_penalty": round(s_tot, 2),
                "ci95_total_penalty": round(ci_tot, 2),
            })

        stat_sum_df = pd.DataFrame(stat_summary)

        # Compute paired differences & Cohen's d effect size relative to QM-ARC
        extended_rows = []
        for prop_str in ["0.5%", "5.0%"]:
            sub = stat_sum_df[stat_sum_df["cache_proportion"] == prop_str].copy()
            qm_m_hp = sub[sub["algorithm"] == "QM-ARC"]["mean_penalty_hp"].iloc[0]
            qm_s_hp = sub[sub["algorithm"] == "QM-ARC"]["std_penalty_hp"].iloc[0]

            for _, row in sub.iterrows():
                r_dict = dict(row)
                algo = r_dict["algorithm"]
                if algo == "QM-ARC":
                    r_dict["hp_diff_pct_paper"] = 0.0
                    r_dict["cohens_d"] = 0.0
                else:
                    # Paper formula: ((P_QM - P_Strat) / P_Strat) * 100
                    p_strat = r_dict["mean_penalty_hp"]
                    r_dict["hp_diff_pct_paper"] = round(((qm_m_hp - p_strat) / p_strat * 100.0), 2)
                    # Cohen's d = (mean_strat - mean_qm) / pooled_std
                    pooled_s = math.sqrt((qm_s_hp ** 2 + r_dict["std_penalty_hp"] ** 2) / 2.0)
                    r_dict["cohens_d"] = round((p_strat - qm_m_hp) / pooled_s, 2) if pooled_s > 0 else 0.0
                extended_rows.append(r_dict)

        final_stat_df = pd.DataFrame(extended_rows)
        final_stat_df.to_csv(self.stat_dir / "statistical_analysis_summary.csv", index=False)
        return final_stat_df

    def aggregate_and_report(self, all_results: List[ExperimentResult]):
        """Consolidate principal results, calculate Table 3, and save aggregated CSVs."""
        records = []
        for r in all_results:
            records.append({
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
            })

        df = pd.DataFrame(records)
        df.to_csv(self.agg_dir / "all_experiments_raw.csv", index=False)

        # Multi-key grouping: strictly separates workloads, request counts, and configs
        group_cols = ["algorithm", "workload", "requests", "num_tiers", "cache_proportion", "penalty_lambda"]
        metric_cols = ["ghr", "hphr", "lphr", "penalty_hp", "penalty_lp", "total_penalty", "avg_latency_ms"]
        agg_mean = df.groupby(group_cols)[metric_cols].mean().reset_index()
        agg_mean.to_csv(self.agg_dir / "experiments_aggregated_mean.csv", index=False)

        # -------------------------------------------------------------
        # Table 3: Percentage difference in penalties relative to Strategy
        # Formula: Diff = ((P_QM - P_Strat) / P_Strat) * 100
        # Negative indicates QM-ARC performed better.
        # -------------------------------------------------------------
        t3_rows = []
        exp1_df = df[(df["num_tiers"] == 2) & (df["penalty_lambda"] == 0.20)]
        for prop in sorted(exp1_df["cache_proportion"].unique()):
            for wl in sorted(exp1_df["workload"].unique()):
                sub = exp1_df[(exp1_df["cache_proportion"] == prop) & (exp1_df["workload"] == wl)]
                qm_sub = sub[sub["algorithm"] == "QM-ARC"]
                if qm_sub.empty:
                    continue
                qm_hp = qm_sub["penalty_hp"].mean()
                qm_lp = qm_sub["penalty_lp"].mean()

                for algo in sorted(sub["algorithm"].unique()):
                    if algo == "QM-ARC":
                        continue
                    a_sub = sub[sub["algorithm"] == algo]
                    a_hp = a_sub["penalty_hp"].mean()
                    a_lp = a_sub["penalty_lp"].mean()

                    diff_hp = ((qm_hp - a_hp) / a_hp * 100.0) if a_hp > 0 else 0.0
                    diff_lp = ((qm_lp - a_lp) / a_lp * 100.0) if a_lp > 0 else 0.0

                    t3_rows.append({
                        "cache_proportion": f"{prop*100:.1f}%",
                        "workload": wl,
                        "strategy": algo,
                        "hp_penalty_diff_pct": round(diff_hp, 2),
                        "lp_penalty_diff_pct": round(diff_lp, 2),
                        "qm_hp_penalty": round(qm_hp, 2),
                        "strategy_hp_penalty": round(a_hp, 2),
                        "qm_lp_penalty": round(qm_lp, 2),
                        "strategy_lp_penalty": round(a_lp, 2),
                    })

        t3_df = pd.DataFrame(t3_rows)
        t3_df.to_csv(self.reports_dir / "table_3_penalty_comparison.csv", index=False)

        # -------------------------------------------------------------
        # Generate Publication Plots
        # -------------------------------------------------------------
        colors = {
            "QM-ARC": "#1f77b4",
            "QL-QM-ARC": "#2ca02c",
            "M-ARC": "#ff7f0e",
            "Priority-LRU": "#d62728",
            "LRU": "#9467bd",
            "LFU": "#8c564b",
            "Random": "#7f7f7f",
        }

        # Exp 1 Plots
        for wl in exp1_df["workload"].unique():
            sub = exp1_df[exp1_df["workload"] == wl]
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            for algo in sorted(sub["algorithm"].unique()):
                a_df = sub[sub["algorithm"] == algo].groupby("cache_proportion")[metric_cols].mean().reset_index().sort_values("cache_proportion")
                c = colors.get(algo, "black")
                p_labels = [f"{p*100:.1f}%" for p in a_df["cache_proportion"]]
                ax1.plot(p_labels, a_df["ghr"], marker="o", label=algo, color=c, linewidth=2)
                ax2.plot(p_labels, a_df["hphr"], marker="s", label=algo, color=c, linewidth=2)
                ax3.plot(p_labels, a_df["lphr"], marker="^", label=algo, color=c, linewidth=2)

            ax1.set_title("Global Hit Rate")
            ax1.set_ylim(0, 1.05)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax2.set_title("High-Priority Hit Rate")
            ax2.set_ylim(0, 1.05)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax3.set_title("Low-Priority Hit Rate")
            ax3.set_ylim(0, 1.05)
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="best", fontsize=8)
            plt.suptitle(f"Exp 1: Hit Rates vs Cache Proportion ({wl})", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"exp1_{wl}_hit_rates.png", dpi=300)
            plt.close(fig)

            # Penalties
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for algo in sorted(sub["algorithm"].unique()):
                a_df = sub[sub["algorithm"] == algo].groupby("cache_proportion")[metric_cols].mean().reset_index().sort_values("cache_proportion")
                c = colors.get(algo, "black")
                p_labels = [f"{p*100:.1f}%" for p in a_df["cache_proportion"]]
                ax1.plot(p_labels, a_df["penalty_hp"], marker="s", label=algo, color=c, linewidth=2)
                ax2.plot(p_labels, a_df["penalty_lp"], marker="o", label=algo, color=c, linewidth=2)

            ax1.set_title("High-Priority Penalty")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="best", fontsize=8)
            ax2.set_title("Low-Priority Penalty")
            ax2.grid(True, linestyle="--", alpha=0.5)
            plt.suptitle(f"Exp 1: QoS Penalties vs Cache Proportion ({wl})", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"exp1_{wl}_penalties.png", dpi=300)
            plt.close(fig)

        # Exp 2 Lambda Sweep Plot
        exp2_df = df[(df["workload"] == "synthetic_zipf_alpha1.0_hp0.2") & (df["num_tiers"] == 2) & (df["cache_proportion"] == 0.05)]
        if not exp2_df.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for algo in sorted(exp2_df["algorithm"].unique()):
                a_df = exp2_df[exp2_df["algorithm"] == algo].groupby("penalty_lambda")[metric_cols].mean().reset_index().sort_values("penalty_lambda")
                c = colors.get(algo, "black")
                l_labels = [f"{l:.1f}" for l in a_df["penalty_lambda"]]
                ax1.plot(l_labels, a_df["total_penalty"], marker="o", label=algo, color=c, linewidth=2)
                ax2.plot(l_labels, a_df["penalty_hp"], marker="s", label=algo, color=c, linewidth=2)
            ax1.set_title("Total Penalty vs Lambda")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="best", fontsize=8)
            ax2.set_title("High-Priority Penalty vs Lambda")
            ax2.grid(True, linestyle="--", alpha=0.5)
            plt.suptitle("Exp 2: Penalty vs Lambda (Zipf alpha=1.0, 5% Cache)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "exp2_synthetic_zipf_lambda_sweep.png", dpi=300)
            plt.close(fig)

        # Exp 3 Three-Tier Plot
        exp3_df = df[df["num_tiers"] == 3]
        if not exp3_df.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for algo in sorted(exp3_df["algorithm"].unique()):
                a_df = exp3_df[exp3_df["algorithm"] == algo].groupby("cache_proportion")[metric_cols].mean().reset_index().sort_values("cache_proportion")
                c = colors.get(algo, "black")
                p_labels = [f"{p*100:.1f}%" for p in a_df["cache_proportion"]]
                ax1.plot(p_labels, a_df["ghr"], marker="o", label=algo, color=c, linewidth=2)
                ax2.plot(p_labels, a_df["total_penalty"], marker="s", label=algo, color=c, linewidth=2)
            ax1.set_title("3-Tier Global Hit Rate")
            ax1.set_ylim(0, 1.05)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="best", fontsize=8)
            ax2.set_title("3-Tier Total Penalty")
            ax2.grid(True, linestyle="--", alpha=0.5)
            plt.suptitle("Exp 3: Three-Tier Architecture DRAM+SSD+NVMe (1:5:25)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "exp3_synthetic_zipf_three_tier.png", dpi=300)
            plt.close(fig)

        print("\nAll baseline aggregations, Table 3 report, and plots generated successfully!")


if __name__ == "__main__":
    suite = FinalBaselineSuite()
    results = suite.run_principal_experiments()
    suite.aggregate_and_report(results)
    stat_df = suite.run_multi_seed_statistical_suite()
    print("\nFinal Baseline Suite finished successfully!")
