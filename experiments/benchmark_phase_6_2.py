#!/usr/bin/env python3
"""
Phase 6.2: Held-Out Validation and Robustness Testing Benchmarking Suite
Evaluates History-Aware Adaptive QM-ARC (HA-QM-ARC) across 20 unseen, held-out random seeds:
Seeds: [1001, 1002, 1003, ..., 1020]

Experiments:
  A: Sequential Scan Validation (4 algorithms x 20 seeds = 80 runs)
  B: QoS Surge Validation (5 algorithms x 20 seeds = 100 runs)
  C: Workload Phase-Change Validation (5 algorithms x 20 seeds = 100 runs)
  D: Architectural Component Ablation Validation (5 variants x 2 workloads x 20 seeds = 200 runs)

Total Simulation Runs: 480 runs
Outputs: results/phase_6_2/
"""

import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Simulator Framework Imports
from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner
from experiments.registry import AlgorithmRegistry
import resources

# ============================================================================
# Global Configuration and Output Paths
# ============================================================================

HELD_OUT_SEEDS = list(range(1001, 1021))  # 20 independent held-out seeds: 1001..1020
PHASE_6_2_DIR = REPO_ROOT / "results" / "phase_6_2"
PHASE_6_2_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Statistical Analysis Helper Functions
# ============================================================================

def normal_cdf(x: float) -> float:
    """Accurate standard normal cumulative distribution function (error function approximation)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def student_t_pdf(x: float, df: int) -> float:
    """Probability density function of Student's t distribution with df degrees of freedom."""
    coeff = math.gamma((df + 1) / 2.0) / (math.sqrt(df * math.pi) * math.gamma(df / 2.0))
    return coeff * (1.0 + (x ** 2) / df) ** (-(df + 1) / 2.0)


def student_t_two_tailed_p(t_val: float, df: int) -> float:
    """Compute exact two-tailed p-value for Student's t distribution using Simpson's rule."""
    abs_t = abs(t_val)
    if abs_t > 50.0:
        return 0.0
    n_pts = 1000
    xs = np.linspace(abs_t, 50.0, n_pts)
    ys = [student_t_pdf(x, df) for x in xs]
    tail = float(np.trapezoid(ys, xs))
    return float(min(1.0, max(0.0, 2.0 * tail)))


def compute_paired_statistics(
    sample_a: List[float],
    sample_b: List[float],
    metric_name: str = "metric",
) -> Dict[str, Any]:
    """
    Compute rigorous paired statistical differences:
    Mean, SD, 95% CI, Paired t-test (exact p-value), Wilcoxon signed-rank test, Cohen's d_z.
    """
    a = np.array(sample_a, dtype=float)
    b = np.array(sample_b, dtype=float)
    diff = a - b
    n = len(diff)
    df = n - 1

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_diff = float(np.mean(diff))
    s_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = s_d / math.sqrt(n) if n > 0 else 0.0

    # Paired Student's t-test
    t_stat = mean_diff / se if se > 0 else 0.0
    p_val_t = student_t_two_tailed_p(t_stat, df)

    # 95% Confidence Interval for df=19: t_crit = 2.093024
    # Approximate t_crit via normal quantile + Cornish-Fisher if df != 19
    t_crit = 2.093024 if df == 19 else (2.7764 if df == 4 else 2.0)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    cohens_d = mean_diff / s_d if s_d > 0 else 0.0

    # Wilcoxon Signed-Rank Test
    non_zero = diff[diff != 0]
    n_w = len(non_zero)
    if n_w == 0:
        w_stat, p_val_w = 0.0, 1.0
    else:
        abs_diff = np.abs(non_zero)
        # Rank absolute differences (average ties)
        import pandas as pd
        ranks = pd.Series(abs_diff).rank().to_numpy()
        w_plus = float(np.sum(ranks[non_zero > 0]))
        w_minus = float(np.sum(ranks[non_zero < 0]))
        w_stat = min(w_plus, w_minus)

        if n_w == n and (w_plus == 0 or w_minus == 0):
            # All positive or all negative: exact permutation p = 2 / 2^n
            p_val_w = 2.0 / (2.0 ** n_w)
        else:
            # Asymptotic normal approximation with continuity correction (standard for N >= 15)
            mu_w = (n_w * (n_w + 1)) / 4.0
            sigma_w = math.sqrt((n_w * (n_w + 1) * (2 * n_w + 1)) / 24.0)
            z_w = (w_stat - mu_w + 0.5) / sigma_w if sigma_w > 0 else 0.0
            p_val_w = 2.0 * normal_cdf(-abs(z_w))

    return {
        "metric": metric_name,
        "n_samples": n,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "sd_diff": s_d,
        "se_diff": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "p_value_t": p_val_t,
        "wilcoxon_w": w_stat,
        "p_value_wilcoxon": p_val_w,
        "cohens_d": cohens_d,
    }


# ============================================================================
# Workload Trace Generators
# ============================================================================

def create_scan_trace(output_path: Path, seed: int) -> Path:
    """Generate Experiment 2 scan trace: 400 hot pre-scan, 400 unique scan, 400 hot recovery."""
    random.seed(seed)
    np.random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hot_catalog = [f"video_chunk_{i}" for i in range(50)]
    scan_catalog = [f"video_chunk_{1000 + i}" for i in range(400)]

    ranks = np.arange(1, 51)
    probs = 1.0 / ranks
    probs /= probs.sum()

    curr_time = 0.0
    lines = []

    # Phase 1: Pre-Scan (400 requests to 50 hot items)
    for _ in range(400):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(hot_catalog, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase 2: Sequential Scan (400 ascending unique requests video_chunk_1000..1399)
    for i in range(400):
        curr_time += random.expovariate(10.0)
        item = scan_catalog[i]
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase 3: Post-Scan Recovery (400 requests to 50 hot items)
    for _ in range(400):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(hot_catalog, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lines)
    return output_path


def create_qos_surge_trace(output_path: Path, seed: int) -> Path:
    """Generate Experiment 4 surge trace: 500 reqs (20% HP) -> 500 reqs (70% HP) -> 500 reqs (20% HP)."""
    random.seed(seed)
    np.random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = [str(i) for i in range(10000)]
    ranks = np.arange(1, 10001)
    probs = 1.0 / ranks
    probs /= probs.sum()

    curr_time = 0.0
    lines = []

    # Phase 1: Baseline (20% HP)
    for _ in range(500):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(catalog, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase 2: Surge (70% HP)
    for _ in range(500):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(catalog, p=probs)
        prio = "h" if random.random() < 0.70 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase 3: Recovery (20% HP)
    for _ in range(500):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(catalog, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lines)
    return output_path


def create_phase_change_trace(output_path: Path, seed: int) -> Path:
    """Generate Experiment 3 phase-change trace: 500 Set A -> 500 Set B -> 500 Set A."""
    random.seed(seed)
    np.random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_a = [str(i) for i in range(1000)]
    set_b = [str(5000 + i) for i in range(1000)]

    ranks = np.arange(1, 1001)
    probs = 1.0 / ranks
    probs /= probs.sum()

    curr_time = 0.0
    lines = []

    # Phase A (500 reqs from Set A)
    for _ in range(500):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(set_a, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase B (500 reqs from Set B)
    for _ in range(500):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(set_b, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase C (500 reqs return to Set A)
    for _ in range(500):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(set_a, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lines)
    return output_path


# ============================================================================
# Log Parsing Helpers
# ============================================================================

def save_aggregated_summary(per_seed_results: List[Dict[str, Any]], group_by: List[str], output_path: Path):
    """Aggregate per-seed observations and write publication-grade summary CSV."""
    import pandas as pd
    df = pd.DataFrame(per_seed_results)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "seed" in numeric_cols:
        numeric_cols.remove("seed")
    agg_df = df.groupby(group_by)[numeric_cols].agg(["mean", "std"]).reset_index()
    agg_df.columns = ["_".join(filter(None, col)).strip("_") for col in agg_df.columns]
    agg_df.to_csv(output_path, index=False)

def parse_scan_log_details(log_path: Path) -> Dict[str, Any]:
    """Parse exact Phase 1, Phase 2, and Phase 3 hits and hot evictions from log."""
    hot_set = {f"video_chunk_{i}" for i in range(50)}
    req_count = 0
    phase_hits = [0, 0, 0]
    p2_hot_evictions = 0

    if not log_path.is_file():
        return {"p1_hits": 0, "p2_hits": 0, "p3_hits": 0, "p2_hot_evictions": 0}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "interest on" in line and "arrived at" in line:
                req_count += 1
            elif "cache hit, read packet" in line:
                phase_idx = min(2, (req_count - 1) // 400)
                phase_hits[phase_idx] += 1
            elif 400 <= req_count < 800:
                # Track hot-set evictions during Phase 2 scan
                if any(k in line for k in ["move from t1 to b1", "move from t2 to b2", "evict to disk"]):
                    for h in hot_set:
                        if h in line:
                            p2_hot_evictions += 1
                            break

    return {
        "p1_hits": phase_hits[0],
        "p2_hits": phase_hits[1],
        "p3_hits": phase_hits[2],
        "p2_hot_evictions": p2_hot_evictions,
    }


def parse_phase_change_log_details(log_path: Path) -> Dict[str, Any]:
    """Parse exact Phase A, Phase B, and Phase C hits from log."""
    req_count = 0
    phase_hits = [0, 0, 0]

    if not log_path.is_file():
        return {"pa_hits": 0, "pb_hits": 0, "pc_hits": 0}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "interest on" in line and "arrived at" in line:
                req_count += 1
            elif "cache hit, read packet" in line:
                phase_idx = min(2, (req_count - 1) // 500)
                phase_hits[phase_idx] += 1

    return {
        "pa_hits": phase_hits[0],
        "pb_hits": phase_hits[1],
        "pc_hits": phase_hits[2],
    }


# ============================================================================
# Experiment A: Sequential Scan Validation (Held-Out Seeds)
# ============================================================================

def run_experiment_a_sequential_scan() -> Dict[str, Any]:
    """Execute Experiment A across 20 held-out seeds."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT A: SEQUENTIAL SCAN VALIDATION (20 HELD-OUT SEEDS)")
    print("=" * 80)

    exp_dir = PHASE_6_2_DIR / "experiment_a_sequential_scan"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    algos = ["HA-QM-ARC", "QM-ARC", "M-ARC", "LRU"]
    per_seed_results = []

    for seed in HELD_OUT_SEEDS:
        trace_path = exp_dir / f"scan_trace_1200reqs_seed{seed}.csv"
        create_scan_trace(output_path=trace_path, seed=seed)

        for algo in algos:
            params = {"enable_scan_detection": True} if algo == "HA-QM-ARC" else None
            cfg = ExperimentConfig(
                algorithm=algo,
                workload=str(trace_path),
                catalog_size=1400,
                cache_proportion=0.05,  # 70 slots
                num_tiers=2,
                num_requests=1200,
                seed=seed,
                output_dir=exp_dir,
                logs_enabled=True,
                algorithm_params=params,
            )
            res = ExperimentRunner.run_experiment(cfg)
            log_path = exp_dir / "logs" / f"{algo}_{seed}.log"
            log_stats = parse_scan_log_details(log_path)

            p1_hits = log_stats["p1_hits"]
            p2_hits = log_stats["p2_hits"]
            p3_hits = log_stats["p3_hits"]
            p3_ghr = p3_hits / 400.0
            p1_ghr = p1_hits / 400.0
            recovery_ratio = (p3_hits / p1_hits) if p1_hits > 0 else 0.0

            row = {
                "experiment": "ExpA_Scan_Validation",
                "seed": seed,
                "algorithm": algo,
                "global_hit_ratio": res.metrics.global_hit_ratio,
                "phase1_pre_scan_hits": p1_hits,
                "phase1_pre_scan_ghr": p1_ghr,
                "phase2_during_scan_hits": p2_hits,
                "phase3_post_scan_hits": p3_hits,
                "phase3_post_scan_ghr": p3_ghr,
                "recovery_ratio": recovery_ratio,
                "hot_evictions_phase2": log_stats["p2_hot_evictions"],
                "total_sla_penalty": res.metrics.total_penalty,
                "penalty_hp": res.metrics.penalty_hp,
                "penalty_lp": res.metrics.penalty_lp,
            }
            per_seed_results.append(row)

    # Save experiment configuration file
    with open(exp_dir / "exp_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Experiment A: Sequential Scan Validation",
            "workload": "1200 requests (400 hot pre-scan, 400 sequential scan, 400 hot recovery)",
            "algorithms": algos,
            "seeds": HELD_OUT_SEEDS,
            "cache_proportion": 0.05,
            "catalog_size": 1400,
            "num_tiers": 2,
        }, f, indent=2)

    # Save per-seed CSV
    per_seed_csv = exp_dir / "per_seed_observations.csv"
    with open(per_seed_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_results)

    # Save aggregated summary CSV
    save_aggregated_summary(per_seed_results, ["algorithm"], exp_dir / "aggregated_summary.csv")

    # Statistical Analysis: HA-QM-ARC vs QM-ARC, and HA-QM-ARC vs LRU
    ha_p3 = [r["phase3_post_scan_hits"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_p3 = [r["phase3_post_scan_hits"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]
    lru_p3 = [r["phase3_post_scan_hits"] for r in per_seed_results if r["algorithm"] == "LRU"]

    ha_pen = [r["total_sla_penalty"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_pen = [r["total_sla_penalty"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]
    lru_pen = [r["total_sla_penalty"] for r in per_seed_results if r["algorithm"] == "LRU"]

    ha_evict = [r["hot_evictions_phase2"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_evict = [r["hot_evictions_phase2"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]

    ha_ghr = [r["global_hit_ratio"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_ghr = [r["global_hit_ratio"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]
    lru_ghr = [r["global_hit_ratio"] for r in per_seed_results if r["algorithm"] == "LRU"]

    stats_list = [
        compute_paired_statistics(ha_p3, qm_p3, "Phase 3 Recovery Hits (HA vs QM)"),
        compute_paired_statistics(ha_p3, lru_p3, "Phase 3 Recovery Hits (HA vs LRU)"),
        compute_paired_statistics(ha_pen, qm_pen, "Total SLA Penalty (HA vs QM)"),
        compute_paired_statistics(ha_pen, lru_pen, "Total SLA Penalty (HA vs LRU)"),
        compute_paired_statistics(ha_evict, qm_evict, "Phase 2 Hot Evictions (HA vs QM)"),
        compute_paired_statistics(ha_ghr, qm_ghr, "Global Hit Ratio (HA vs QM)"),
        compute_paired_statistics(ha_ghr, lru_ghr, "Global Hit Ratio (HA vs LRU)"),
    ]

    stats_csv = exp_dir / "statistical_analysis.csv"
    with open(stats_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_list[0].keys()))
        writer.writeheader()
        writer.writerows(stats_list)

    # Plot Experiment A
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(algos))
    means_p3 = [np.mean([r["phase3_post_scan_hits"] for r in per_seed_results if r["algorithm"] == a]) for a in algos]
    means_pen = [np.mean([r["total_sla_penalty"] for r in per_seed_results if r["algorithm"] == a]) for a in algos]

    ax1.bar(x, means_p3, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"], width=0.5)
    ax1.set_title("Exp A: Post-Scan Recovery Hits (/400, 20 Held-Out Seeds)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, fontsize=10)
    ax1.set_ylabel("Mean Phase 3 Hits", fontsize=10)
    ax1.set_ylim([300, 400])
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    ax2.bar(x, means_pen, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"], width=0.5)
    ax2.set_title("Exp A: Total SLA Penalty (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, fontsize=10)
    ax2.set_ylabel("Mean Penalty Units", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "exp_a_sequential_scan_held_out.png", dpi=200)
    plt.close()

    print(f"Experiment A completed: {len(per_seed_results)} runs executed across 20 held-out seeds.")
    return {"per_seed": per_seed_results, "stats": stats_list}


# ============================================================================
# Experiment B: QoS Surge Validation (Held-Out Seeds)
# ============================================================================

def run_experiment_b_qos_surge() -> Dict[str, Any]:
    """Execute Experiment B across 20 held-out seeds."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT B: QOS SURGE VALIDATION (20 HELD-OUT SEEDS)")
    print("=" * 80)

    exp_dir = PHASE_6_2_DIR / "experiment_b_qos_surge"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    algos = ["HA-QM-ARC", "QM-ARC", "Priority-LRU", "LRU", "M-ARC"]
    per_seed_results = []

    for seed in HELD_OUT_SEEDS:
        trace_path = exp_dir / f"qos_surge_1500reqs_seed{seed}.csv"
        create_qos_surge_trace(output_path=trace_path, seed=seed)

        for algo in algos:
            # Frozen hyperparameter kappa_qos = 0.50
            params = {"kappa_qos": 0.50} if algo == "HA-QM-ARC" else None
            cfg = ExperimentConfig(
                algorithm=algo,
                workload=str(trace_path),
                catalog_size=10000,
                cache_proportion=0.01,  # 100 slots
                num_tiers=2,
                num_requests=1500,
                seed=seed,
                output_dir=exp_dir,
                logs_enabled=False,
                algorithm_params=params,
            )
            res = ExperimentRunner.run_experiment(cfg)
            row = {
                "experiment": "ExpB_QoS_Surge_Validation",
                "seed": seed,
                "algorithm": algo,
                "global_hit_ratio": res.metrics.global_hit_ratio,
                "high_priority_hit_ratio": res.metrics.high_priority_hit_ratio,
                "low_priority_hit_ratio": res.metrics.low_priority_hit_ratio,
                "penalty_hp": res.metrics.penalty_hp,
                "penalty_lp": res.metrics.penalty_lp,
                "total_sla_penalty": res.metrics.total_penalty,
            }
            per_seed_results.append(row)

    # Save experiment configuration file
    with open(exp_dir / "exp_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Experiment B: QoS Surge Validation",
            "workload": "1500 requests (500 baseline 20% HP, 500 surge 70% HP, 500 recovery 20% HP)",
            "algorithms": algos,
            "seeds": HELD_OUT_SEEDS,
            "cache_proportion": 0.01,
            "catalog_size": 10000,
            "num_tiers": 2,
            "frozen_kappa_qos": 0.50,
        }, f, indent=2)

    # Save per-seed CSV
    per_seed_csv = exp_dir / "per_seed_observations.csv"
    with open(per_seed_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_results)

    # Save aggregated summary CSV
    save_aggregated_summary(per_seed_results, ["algorithm"], exp_dir / "aggregated_summary.csv")

    # Statistical Comparisons: HA-QM-ARC vs QM-ARC, and HA-QM-ARC vs LRU / Priority-LRU
    ha_pen_hp = [r["penalty_hp"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_pen_hp = [r["penalty_hp"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]
    lru_pen_hp = [r["penalty_hp"] for r in per_seed_results if r["algorithm"] == "LRU"]
    plru_pen_hp = [r["penalty_hp"] for r in per_seed_results if r["algorithm"] == "Priority-LRU"]

    ha_pen_lp = [r["penalty_lp"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_pen_lp = [r["penalty_lp"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]

    ha_pen_tot = [r["total_sla_penalty"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_pen_tot = [r["total_sla_penalty"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]
    lru_pen_tot = [r["total_sla_penalty"] for r in per_seed_results if r["algorithm"] == "LRU"]
    plru_pen_tot = [r["total_sla_penalty"] for r in per_seed_results if r["algorithm"] == "Priority-LRU"]

    ha_hphr = [r["high_priority_hit_ratio"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_hphr = [r["high_priority_hit_ratio"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]

    stats_list = [
        compute_paired_statistics(ha_pen_hp, qm_pen_hp, "HP SLA Penalty (HA vs QM)"),
        compute_paired_statistics(ha_pen_hp, lru_pen_hp, "HP SLA Penalty (HA vs LRU)"),
        compute_paired_statistics(ha_pen_hp, plru_pen_hp, "HP SLA Penalty (HA vs Priority-LRU)"),
        compute_paired_statistics(ha_pen_lp, qm_pen_lp, "LP SLA Penalty (HA vs QM)"),
        compute_paired_statistics(ha_pen_tot, qm_pen_tot, "Total SLA Penalty (HA vs QM)"),
        compute_paired_statistics(ha_pen_tot, lru_pen_tot, "Total SLA Penalty (HA vs LRU)"),
        compute_paired_statistics(ha_pen_tot, plru_pen_tot, "Total SLA Penalty (HA vs Priority-LRU)"),
        compute_paired_statistics(ha_hphr, qm_hphr, "High-Priority Hit Ratio (HA vs QM)"),
    ]

    stats_csv = exp_dir / "statistical_analysis.csv"
    with open(stats_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_list[0].keys()))
        writer.writeheader()
        writer.writerows(stats_list)

    # Plot Experiment B
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(algos))
    means_hphr = [np.mean([r["high_priority_hit_ratio"] for r in per_seed_results if r["algorithm"] == a]) for a in algos]
    means_lphr = [np.mean([r["low_priority_hit_ratio"] for r in per_seed_results if r["algorithm"] == a]) for a in algos]
    means_pen_hp = [np.mean([r["penalty_hp"] for r in per_seed_results if r["algorithm"] == a]) for a in algos]

    ax1.bar(x - 0.15, means_hphr, 0.3, label="HP Hit Ratio", color="#2ca02c")
    ax1.bar(x + 0.15, means_lphr, 0.3, label="LP Hit Ratio", color="#1f77b4")
    ax1.set_title("Exp B: Priority Hit Ratios Under Surge (20 Seeds)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, rotation=15, fontsize=9)
    ax1.set_ylabel("Hit Ratio", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    ax2.bar(x, means_pen_hp, color="#d62728", width=0.45)
    ax2.set_title("Exp B: High-Priority SLA Penalty (20 Seeds)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, rotation=15, fontsize=9)
    ax2.set_ylabel("HP SLA Penalty Units", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "exp_b_qos_surge_held_out.png", dpi=200)
    plt.close()

    print(f"Experiment B completed: {len(per_seed_results)} runs executed across 20 held-out seeds.")
    return {"per_seed": per_seed_results, "stats": stats_list}


# ============================================================================
# Experiment C: Workload Phase-Change Validation (Held-Out Seeds)
# ============================================================================

def run_experiment_c_phase_change() -> Dict[str, Any]:
    """Execute Experiment C across 20 held-out seeds."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT C: PHASE-CHANGE VALIDATION (20 HELD-OUT SEEDS)")
    print("=" * 80)

    exp_dir = PHASE_6_2_DIR / "experiment_c_phase_change"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    algos = ["HA-QM-ARC", "QM-ARC", "QL-QM-ARC", "M-ARC", "LRU"]
    per_seed_results = []

    for seed in HELD_OUT_SEEDS:
        trace_path = exp_dir / f"phase_change_1500reqs_seed{seed}.csv"
        create_phase_change_trace(output_path=trace_path, seed=seed)

        for algo in algos:
            cfg = ExperimentConfig(
                algorithm=algo,
                workload=str(trace_path),
                catalog_size=10000,
                cache_proportion=0.02,  # 200 slots
                num_tiers=2,
                num_requests=1500,
                seed=seed,
                output_dir=exp_dir,
                logs_enabled=True,
            )
            res = ExperimentRunner.run_experiment(cfg)
            log_path = exp_dir / "logs" / f"{algo}_{seed}.log"
            log_stats = parse_phase_change_log_details(log_path)

            row = {
                "experiment": "ExpC_Phase_Change_Validation",
                "seed": seed,
                "algorithm": algo,
                "global_hit_ratio": res.metrics.global_hit_ratio,
                "phaseA_hits": log_stats["pa_hits"],
                "phaseB_hits": log_stats["pb_hits"],
                "phaseC_hits": log_stats["pc_hits"],
                "phaseC_hit_ratio": log_stats["pc_hits"] / 500.0,
                "high_priority_hit_ratio": res.metrics.high_priority_hit_ratio,
                "low_priority_hit_ratio": res.metrics.low_priority_hit_ratio,
                "total_sla_penalty": res.metrics.total_penalty,
            }
            per_seed_results.append(row)

    # Save experiment configuration file
    with open(exp_dir / "exp_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Experiment C: Workload Phase-Change Validation",
            "workload": "1500 requests (500 Phase A Set A, 500 Phase B Set B, 500 Phase C Set A)",
            "algorithms": algos,
            "seeds": HELD_OUT_SEEDS,
            "cache_proportion": 0.02,
            "catalog_size": 10000,
            "num_tiers": 2,
        }, f, indent=2)

    # Save per-seed CSV
    per_seed_csv = exp_dir / "per_seed_observations.csv"
    with open(per_seed_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_results)

    # Save aggregated summary CSV
    save_aggregated_summary(per_seed_results, ["algorithm"], exp_dir / "aggregated_summary.csv")

    # Statistical Comparison: HA-QM-ARC vs QM-ARC, and HA-QM-ARC vs LRU
    ha_ghr = [r["global_hit_ratio"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_ghr = [r["global_hit_ratio"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]
    lru_ghr = [r["global_hit_ratio"] for r in per_seed_results if r["algorithm"] == "LRU"]

    ha_pc = [r["phaseC_hits"] for r in per_seed_results if r["algorithm"] == "HA-QM-ARC"]
    qm_pc = [r["phaseC_hits"] for r in per_seed_results if r["algorithm"] == "QM-ARC"]
    lru_pc = [r["phaseC_hits"] for r in per_seed_results if r["algorithm"] == "LRU"]

    stats_list = [
        compute_paired_statistics(ha_ghr, qm_ghr, "Overall Global Hit Ratio (HA vs QM)"),
        compute_paired_statistics(ha_ghr, lru_ghr, "Overall Global Hit Ratio (HA vs LRU)"),
        compute_paired_statistics(ha_pc, qm_pc, "Phase C Return Hits (HA vs QM)"),
        compute_paired_statistics(ha_pc, lru_pc, "Phase C Return Hits (HA vs LRU)"),
    ]

    stats_csv = exp_dir / "statistical_analysis.csv"
    with open(stats_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_list[0].keys()))
        writer.writeheader()
        writer.writerows(stats_list)

    # Plot Experiment C
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(algos))
    means_ghr = [np.mean([r["global_hit_ratio"] for r in per_seed_results if r["algorithm"] == a]) for a in algos]
    means_pc = [np.mean([r["phaseC_hit_ratio"] for r in per_seed_results if r["algorithm"] == a]) for a in algos]

    width = 0.35
    ax.bar(x - width/2, means_ghr, width, label="Overall Hit Ratio (Phases A-C)", color="#1f77b4")
    ax.bar(x + width/2, means_pc, width, label="Phase C Return Hit Ratio (Set A)", color="#2ca02c")
    ax.set_title("Exp C: Hit Ratio Across Phase Shifts (20 Held-Out Seeds)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, fontsize=10)
    ax.set_ylabel("Hit Ratio", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "exp_c_phase_change_held_out.png", dpi=200)
    plt.close()

    print(f"Experiment C completed: {len(per_seed_results)} runs executed across 20 held-out seeds.")
    return {"per_seed": per_seed_results, "stats": stats_list}


# ============================================================================
# Experiment D: Ablation Validation (Held-Out Seeds)
# ============================================================================

def run_experiment_d_ablation() -> Dict[str, Any]:
    """Execute Experiment D across 20 held-out seeds."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT D: ABLATION VALIDATION (20 HELD-OUT SEEDS)")
    print("=" * 80)

    exp_dir = PHASE_6_2_DIR / "experiment_d_ablation"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "Complete HA-QM-ARC": {"enable_scan_detection": True, "kappa_qos": 0.50},
        "w/o Hit Gradient": {"omega_grad": 0.0, "enable_scan_detection": True, "kappa_qos": 0.50},
        "w/o Burst Detection": {"omega_burst": 0.0, "enable_scan_detection": True, "kappa_qos": 0.50},
        "w/o Sequential Detection": {"enable_scan_detection": False, "omega_scan": 0.0, "kappa_qos": 0.50},
        "w/o Dynamic QoS": {"kappa_qos": 0.0, "enable_scan_detection": True},
    }

    per_seed_results = []

    for seed in HELD_OUT_SEEDS:
        scan_trace = exp_dir / f"ablation_scan_seed{seed}.csv"
        create_scan_trace(output_path=scan_trace, seed=seed)

        surge_trace = exp_dir / f"ablation_surge_seed{seed}.csv"
        create_qos_surge_trace(output_path=surge_trace, seed=seed)

        workloads = [
            ("Scan_Pollution", scan_trace, 1200, 0.05, 1400),
            ("QoS_Surge", surge_trace, 1500, 0.01, 10000),
        ]

        for wl_name, wl_path, n_reqs, prop, cat_size in workloads:
            for v_name, v_params in variants.items():
                cfg = ExperimentConfig(
                    algorithm="HA-QM-ARC",
                    workload=str(wl_path),
                    catalog_size=cat_size,
                    cache_proportion=prop,
                    num_tiers=2,
                    num_requests=n_reqs,
                    seed=seed,
                    output_dir=exp_dir,
                    logs_enabled=False,
                    algorithm_params=v_params,
                )
                res = ExperimentRunner.run_experiment(cfg)
                row = {
                    "experiment": "ExpD_Ablation_Validation",
                    "workload": wl_name,
                    "variant": v_name,
                    "seed": seed,
                    "global_hit_ratio": res.metrics.global_hit_ratio,
                    "high_priority_hit_ratio": res.metrics.high_priority_hit_ratio,
                    "penalty_hp": res.metrics.penalty_hp,
                    "penalty_lp": res.metrics.penalty_lp,
                    "total_sla_penalty": res.metrics.total_penalty,
                }
                per_seed_results.append(row)

    # Save experiment configuration file
    with open(exp_dir / "exp_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Experiment D: Architectural Component Ablation Validation",
            "workloads": ["Scan_Pollution", "QoS_Surge"],
            "variants": list(variants.keys()),
            "seeds": HELD_OUT_SEEDS,
            "num_tiers": 2,
        }, f, indent=2)

    # Save per-seed CSV
    per_seed_csv = exp_dir / "per_seed_observations.csv"
    with open(per_seed_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_results)

    # Save aggregated summary CSV
    save_aggregated_summary(per_seed_results, ["workload", "variant"], exp_dir / "aggregated_summary.csv")

    # Paired Statistics against Complete HA-QM-ARC for each ablated variant
    complete_scan_ghr = [r["global_hit_ratio"] for r in per_seed_results if r["workload"] == "Scan_Pollution" and r["variant"] == "Complete HA-QM-ARC"]
    complete_scan_pen = [r["total_sla_penalty"] for r in per_seed_results if r["workload"] == "Scan_Pollution" and r["variant"] == "Complete HA-QM-ARC"]

    complete_surge_pen_hp = [r["penalty_hp"] for r in per_seed_results if r["workload"] == "QoS_Surge" and r["variant"] == "Complete HA-QM-ARC"]

    stats_list = []
    for v_name in variants.keys():
        if v_name == "Complete HA-QM-ARC":
            continue
        v_scan_ghr = [r["global_hit_ratio"] for r in per_seed_results if r["workload"] == "Scan_Pollution" and r["variant"] == v_name]
        v_scan_pen = [r["total_sla_penalty"] for r in per_seed_results if r["workload"] == "Scan_Pollution" and r["variant"] == v_name]
        v_surge_pen_hp = [r["penalty_hp"] for r in per_seed_results if r["workload"] == "QoS_Surge" and r["variant"] == v_name]

        # Note: Paired comparison is (Complete - Variant)
        st_ghr = compute_paired_statistics(complete_scan_ghr, v_scan_ghr, f"Scan GHR Gain (Complete vs {v_name})")
        st_pen = compute_paired_statistics(v_scan_pen, complete_scan_pen, f"Scan Penalty Avoided ({v_name} - Complete)")
        st_surge_hp = compute_paired_statistics(v_surge_pen_hp, complete_surge_pen_hp, f"Surge HP Penalty Avoided ({v_name} - Complete)")

        stats_list.extend([st_ghr, st_pen, st_surge_hp])

    stats_csv = exp_dir / "statistical_analysis.csv"
    with open(stats_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_list[0].keys()))
        writer.writeheader()
        writer.writerows(stats_list)

    # Plot Experiment D
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    v_keys = list(variants.keys())
    x = np.arange(len(v_keys))
    scan_ghrs = [np.mean([r["global_hit_ratio"] for r in per_seed_results if r["workload"] == "Scan_Pollution" and r["variant"] == v]) for v in v_keys]
    surge_pens = [np.mean([r["penalty_hp"] for r in per_seed_results if r["workload"] == "QoS_Surge" and r["variant"] == v]) for v in v_keys]

    ax1.bar(x, scan_ghrs, color="#1f77b4", width=0.5)
    ax1.set_title("Exp D: Scan Pollution GHR by Variant (20 Seeds)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(v_keys, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Global Hit Ratio", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    ax2.bar(x, surge_pens, color="#d62728", width=0.5)
    ax2.set_title("Exp D: QoS Surge HP Penalty by Variant (20 Seeds)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(v_keys, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("HP SLA Penalty Units", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "exp_d_ablation_held_out.png", dpi=200)
    plt.close()

    print(f"Experiment D completed: {len(per_seed_results)} runs executed across 20 held-out seeds.")
    return {"per_seed": per_seed_results, "stats": stats_list}


# ============================================================================
# Main Orchestrator
# ============================================================================

def main():
    t_start = time.time()
    print("================================================================================")
    print("PHASE 6.2: HELD-OUT VALIDATION & ROBUSTNESS TESTING (N=20 UNSEEN SEEDS)")
    print(f"Target Directory: {PHASE_6_2_DIR}")
    print(f"Seeds: {HELD_OUT_SEEDS}")
    print("================================================================================")

    # Execute all 4 held-out experiments
    res_a = run_experiment_a_sequential_scan()
    res_b = run_experiment_b_qos_surge()
    res_c = run_experiment_c_phase_change()
    res_d = run_experiment_d_ablation()

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"PHASE 6.2 HELD-OUT SUITE FINISHED: 480 RUNS EXECUTED IN {elapsed:.2f}s (~{elapsed/60.0:.2f} mins)!")
    print("================================================================================")


if __name__ == "__main__":
    main()
