"""Phase 6 Comprehensive Comparative Experimental Evaluation Suite.

Executes:
1. Experiment 1: Stable Zipf Workload (8 algorithms, 4 cache sizes, 2-tier & 3-tier, 5 seeds).
2. Experiment 2: Sequential Scan Pollution (hot-set hit ratios before/during/after scan).
3. Experiment 3: Sudden Workload Phase Change (multi-phase popularity shift adaptation).
4. Experiment 4: High-Priority Traffic Surge (QoS stress and dynamic lambda attenuation).
5. Experiment 5: Ablation Study (component attribution across 5 variants).
6. Experiment 6: History Window & Parameter Sensitivity (W and weight sweeps).
"""

import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Any, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from common.packet import Packet
from common.zipf import zipf_distribution
from experiments.config import ExperimentConfig, TierConfig
from experiments.registry import AlgorithmRegistry
from experiments.runner import ExperimentRunner
from experiments.results import ExperimentResult
import resources

# Standard experimental seeds
SEEDS = [42, 123, 456, 789, 999]
PHASE_6_RESULTS_DIR = REPO_ROOT / "results" / "phase_6"


# ============================================================================
# Statistical Analysis Helper Functions
# ============================================================================

def student_t_pvalue(t_stat: float, df: int = 4) -> float:
    """Compute two-tailed p-value for Student's t distribution with given df."""
    if df <= 0 or math.isnan(t_stat):
        return 1.0
    t = abs(float(t_stat))
    if t == 0.0:
        return 1.0
    coeff = math.gamma((df + 1) / 2.0) / (math.sqrt(df * math.pi) * math.gamma(df / 2.0))
    def pdf(x):
        return coeff * (1.0 + x * x / df) ** (-(df + 1.0) / 2.0)

    # Numerical integration using Simpson's rule
    N = 2000
    b = max(100.0, t + 50.0)
    h = (b - t) / N
    integral = pdf(t) + pdf(b)
    for i in range(1, N):
        x = t + i * h
        integral += 4 * pdf(x) if (i % 2 == 1) else 2 * pdf(x)
    integral *= (h / 3.0)
    return min(1.0, max(0.0, 2.0 * integral))


def compute_paired_statistics(
    sample_a: List[float],
    sample_b: List[float],
    t_crit_95: float = 2.776,  # df = 4, alpha = 0.05
) -> Dict[str, Any]:
    """Compute paired differences, mean, SD, 95% CI, paired t-test, and Cohen's d."""
    n = len(sample_a)
    assert n == len(sample_b) and n >= 2

    a_arr = np.array(sample_a, dtype=float)
    b_arr = np.array(sample_b, dtype=float)
    diff = a_arr - b_arr

    mean_a = float(np.mean(a_arr))
    sd_a = float(np.std(a_arr, ddof=1)) if n > 1 else 0.0

    mean_b = float(np.mean(b_arr))
    sd_b = float(np.std(b_arr, ddof=1)) if n > 1 else 0.0

    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_diff = sd_diff / math.sqrt(n) if n > 0 else 0.0

    ci_lower = mean_diff - t_crit_95 * se_diff
    ci_upper = mean_diff + t_crit_95 * se_diff

    if sd_diff > 1e-12:
        t_stat = mean_diff / se_diff
        cohen_d = mean_diff / sd_diff
    else:
        t_stat = 0.0
        cohen_d = 0.0

    p_val = student_t_pvalue(t_stat, df=n - 1)

    return {
        "n": n,
        "mean_a": mean_a,
        "sd_a": sd_a,
        "mean_b": mean_b,
        "sd_b": sd_b,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "p_value": p_val,
        "cohens_d": cohen_d,
        "statistically_significant": bool(p_val < 0.05),
    }


# ============================================================================
# Specialized Workload Trace Generators
# ============================================================================

def create_scan_pollution_trace(
    output_path: Path,
    num_hot_items: int = 50,
    scan_unique_items: int = 400,
    phase_len: int = 400,
    seed: int = 42,
) -> Path:
    """Generate a 3-phase trace: Pre-Scan Hot -> Sequential Scan -> Post-Scan Hot."""
    random.seed(seed)
    np.random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hot_catalog = [f"video_chunk_{i}" for i in range(num_hot_items)]
    scan_catalog = [f"video_chunk_{1000 + i}" for i in range(scan_unique_items)]

    ranks = np.arange(1, num_hot_items + 1)
    probs = 1.0 / ranks
    probs /= probs.sum()

    curr_time = 0.0
    lines = []

    # Phase 1: Pre-Scan (400 requests to resident hot items)
    for _ in range(phase_len):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(hot_catalog, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase 2: Sequential Scan (400 requests strictly ascending chunk_1000..chunk_1399)
    for i in range(phase_len):
        curr_time += random.expovariate(10.0)
        item = scan_catalog[i % len(scan_catalog)]
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase 3: Post-Scan Recovery (400 requests return to hot items)
    for _ in range(phase_len):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(hot_catalog, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lines)

    return output_path


def create_phase_change_trace(
    output_path: Path,
    phase_len: int = 500,
    seed: int = 42,
) -> Path:
    """Generate a 3-phase trace: Set A (0..999) -> Set B (5000..5999) -> Set A (0..999)."""
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
    for _ in range(phase_len):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(set_a, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase B (500 reqs from Set B)
    for _ in range(phase_len):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(set_b, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase C (500 reqs return to Set A)
    for _ in range(phase_len):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(set_a, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lines)

    return output_path


def create_qos_surge_trace(
    output_path: Path,
    phase_len: int = 500,
    seed: int = 42,
) -> Path:
    """Generate a 3-phase trace: 20% HP -> 70% HP Surge -> 20% HP Recovery."""
    random.seed(seed)
    np.random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = [str(i) for i in range(10000)]
    ranks = np.arange(1, 10001)
    probs = 1.0 / ranks
    probs /= probs.sum()

    curr_time = 0.0
    lines = []

    # Phase 1: Normal (20% HP)
    for _ in range(phase_len):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(catalog, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase 2: Surge (70% HP)
    for _ in range(phase_len):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(catalog, p=probs)
        prio = "h" if random.random() < 0.70 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    # Phase 3: Recovery (20% HP)
    for _ in range(phase_len):
        curr_time += random.expovariate(10.0)
        item = np.random.choice(catalog, p=probs)
        prio = "h" if random.random() < 0.20 else "l"
        lines.append(["d", f"{curr_time:.6f}", item, 1024, prio, 4.0, 0.05])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lines)

    return output_path


# ============================================================================
# Phase Metric Parsing Helper
# ============================================================================

def parse_phase_metrics_from_log(log_path: Path, phase_boundaries: List[int]) -> List[Dict[str, Any]]:
    """Parse phase-by-phase hit ratios and stats from simulator log file."""
    if not log_path.is_file():
        return []

    phases_data = []
    current_phase_idx = 0
    req_count = 0
    phase_hits = 0
    phase_hp_hits = 0
    phase_lp_hits = 0
    phase_hp_reqs = 0
    phase_lp_reqs = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "interest on" in line and "arrived at" in line:
                req_count += 1
                if current_phase_idx < len(phase_boundaries) and req_count > phase_boundaries[current_phase_idx]:
                    total_p_reqs = phase_hp_reqs + phase_lp_reqs
                    ghr = phase_hits / float(total_p_reqs) if total_p_reqs > 0 else 0.0
                    hphr = phase_hp_hits / float(phase_hp_reqs) if phase_hp_reqs > 0 else 0.0
                    lphr = phase_lp_hits / float(phase_lp_reqs) if phase_lp_reqs > 0 else 0.0
                    phases_data.append({
                        "phase": current_phase_idx + 1,
                        "requests": total_p_reqs,
                        "ghr": ghr,
                        "hphr": hphr,
                        "lphr": lphr,
                    })
                    current_phase_idx += 1
                    phase_hits = phase_hp_hits = phase_lp_hits = 0
                    phase_hp_reqs = phase_lp_reqs = 0

            elif "cache hit, read packet" in line:
                phase_hits += 1

    total_p_reqs = phase_hp_reqs + phase_lp_reqs
    if total_p_reqs > 0 or len(phases_data) < len(phase_boundaries):
        ghr = phase_hits / float(max(1, total_p_reqs))
        phases_data.append({
            "phase": current_phase_idx + 1,
            "requests": total_p_reqs,
            "ghr": ghr,
            "hphr": 0.0,
            "lphr": 0.0,
        })

    return phases_data


# ============================================================================
# Experiment 1: Stable Zipf Workload
# ============================================================================

def run_experiment_1_stable_zipf() -> Dict[str, Any]:
    """Execute Experiment 1: Stable Zipf workload across 8 algorithms, 4 capacities, 2-tier and 3-tier."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT 1: STABLE ZIPF WORKLOAD")
    print("=" * 80)

    exp_dir = PHASE_6_RESULTS_DIR / "experiment_1_stable_zipf"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    algorithms = AlgorithmRegistry.list_algorithms()
    proportions_2tier = [0.005, 0.01, 0.05, 0.10]
    proportions_3tier = [0.01, 0.05, 0.10]

    raw_results: List[Dict[str, Any]] = []

    # 1. Two-Tier Evaluation
    for prop in proportions_2tier:
        for seed in SEEDS:
            # Generate shared workload once per seed
            trace_path = resources.SYNTHETIC_DIR / f"synth_items10000_reqs1000_seed{seed}.csv"
            if not trace_path.is_file():
                from traces.trace_creating_and_parsing.synthetic_trace import generate_synthetic_trace
                generate_synthetic_trace(
                    n_unique_items=10000, zipf_alpha=1.0, high_priority_ratio=0.20,
                    num_requests=1000, seed=seed, output_path=trace_path,
                )

            for algo in algorithms:
                cfg = ExperimentConfig(
                    algorithm=algo,
                    workload=str(trace_path),
                    catalog_size=10000,
                    cache_proportion=prop,
                    num_tiers=2,
                    num_requests=1000,
                    seed=seed,
                    output_dir=exp_dir,
                    logs_enabled=False,
                )
                res = ExperimentRunner.run_experiment(cfg)
                row = {
                    "experiment": "Exp1_Stable_Zipf",
                    "num_tiers": 2,
                    "cache_proportion": prop,
                    "seed": seed,
                    "algorithm": algo,
                    "ghr": res.metrics.global_hit_ratio,
                    "hphr": res.metrics.high_priority_hit_ratio,
                    "lphr": res.metrics.low_priority_hit_ratio,
                    "total_penalty": res.metrics.total_penalty,
                    "penalty_hp": res.metrics.penalty_hp,
                    "penalty_lp": res.metrics.penalty_lp,
                    "avg_retrieval_latency": res.metrics.avg_retrieval_latency,
                    "wall_time": res.metrics.simulation_wall_time_seconds,
                }
                raw_results.append(row)

    # 2. Three-Tier Evaluation
    for prop in proportions_3tier:
        for seed in SEEDS:
            trace_path = resources.SYNTHETIC_DIR / f"synth_items10000_reqs1000_seed{seed}.csv"
            for algo in algorithms:
                cfg = ExperimentConfig(
                    algorithm=algo,
                    workload=str(trace_path),
                    catalog_size=10000,
                    cache_proportion=prop,
                    num_tiers=3,
                    num_requests=1000,
                    seed=seed,
                    output_dir=exp_dir,
                    logs_enabled=False,
                )
                res = ExperimentRunner.run_experiment(cfg)
                row = {
                    "experiment": "Exp1_Stable_Zipf",
                    "num_tiers": 3,
                    "cache_proportion": prop,
                    "seed": seed,
                    "algorithm": algo,
                    "ghr": res.metrics.global_hit_ratio,
                    "hphr": res.metrics.high_priority_hit_ratio,
                    "lphr": res.metrics.low_priority_hit_ratio,
                    "total_penalty": res.metrics.total_penalty,
                    "penalty_hp": res.metrics.penalty_hp,
                    "penalty_lp": res.metrics.penalty_lp,
                    "avg_retrieval_latency": res.metrics.avg_retrieval_latency,
                    "wall_time": res.metrics.simulation_wall_time_seconds,
                }
                raw_results.append(row)

    # Save Aggregated Summary CSV
    csv_path = exp_dir / "aggregated_summary.csv"
    if raw_results:
        keys = list(raw_results[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(raw_results)

    # Statistical Analysis: Compare HA-QM-ARC against QM-ARC, QL-QM-ARC, and LRU
    stats_rows = []
    for num_tiers in [2, 3]:
        props = proportions_2tier if num_tiers == 2 else proportions_3tier
        for prop in props:
            ha_ghr = [r["ghr"] for r in raw_results if r["num_tiers"] == num_tiers and r["cache_proportion"] == prop and r["algorithm"] == "HA-QM-ARC"]
            qm_ghr = [r["ghr"] for r in raw_results if r["num_tiers"] == num_tiers and r["cache_proportion"] == prop and r["algorithm"] == "QM-ARC"]
            ql_ghr = [r["ghr"] for r in raw_results if r["num_tiers"] == num_tiers and r["cache_proportion"] == prop and r["algorithm"] == "QL-QM-ARC"]
            lru_ghr = [r["ghr"] for r in raw_results if r["num_tiers"] == num_tiers and r["cache_proportion"] == prop and r["algorithm"] == "LRU"]

            ha_pen = [r["total_penalty"] for r in raw_results if r["num_tiers"] == num_tiers and r["cache_proportion"] == prop and r["algorithm"] == "HA-QM-ARC"]
            qm_pen = [r["total_penalty"] for r in raw_results if r["num_tiers"] == num_tiers and r["cache_proportion"] == prop and r["algorithm"] == "QM-ARC"]

            st_qm = compute_paired_statistics(ha_ghr, qm_ghr)
            st_ql = compute_paired_statistics(ha_ghr, ql_ghr)
            st_lru = compute_paired_statistics(ha_ghr, lru_ghr)
            st_pen = compute_paired_statistics(ha_pen, qm_pen)

            stats_rows.append({
                "num_tiers": num_tiers, "cache_proportion": prop, "comparison": "HA-QM-ARC vs QM-ARC (GHR)",
                "mean_ha": st_qm["mean_a"], "mean_baseline": st_qm["mean_b"], "diff_mean": st_qm["mean_diff"],
                "ci_95": f"[{st_qm['ci_lower']:.4f}, {st_qm['ci_upper']:.4f}]", "t_stat": st_qm["t_stat"],
                "p_value": st_qm["p_value"], "cohens_d": st_qm["cohens_d"],
            })
            stats_rows.append({
                "num_tiers": num_tiers, "cache_proportion": prop, "comparison": "HA-QM-ARC vs QL-QM-ARC (GHR)",
                "mean_ha": st_ql["mean_a"], "mean_baseline": st_ql["mean_b"], "diff_mean": st_ql["mean_diff"],
                "ci_95": f"[{st_ql['ci_lower']:.4f}, {st_ql['ci_upper']:.4f}]", "t_stat": st_ql["t_stat"],
                "p_value": st_ql["p_value"], "cohens_d": st_ql["cohens_d"],
            })
            stats_rows.append({
                "num_tiers": num_tiers, "cache_proportion": prop, "comparison": "HA-QM-ARC vs QM-ARC (Penalty)",
                "mean_ha": st_pen["mean_a"], "mean_baseline": st_pen["mean_b"], "diff_mean": st_pen["mean_diff"],
                "ci_95": f"[{st_pen['ci_lower']:.2f}, {st_pen['ci_upper']:.2f}]", "t_stat": st_pen["t_stat"],
                "p_value": st_pen["p_value"], "cohens_d": st_pen["cohens_d"],
            })

    stats_csv = exp_dir / "statistical_analysis.csv"
    if stats_rows:
        with open(stats_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)

    # Generate Plots
    _plot_exp1_results(raw_results, plots_dir)

    print(f"Experiment 1 complete: {len(raw_results)} runs executed. Outputs saved to {exp_dir}")
    return {"raw_results": raw_results, "stats": stats_rows}


def _plot_exp1_results(raw_results: List[Dict[str, Any]], plots_dir: Path):
    """Generate publication-quality plots for Experiment 1."""
    # 2-Tier GHR Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    algos = ["HA-QM-ARC", "QM-ARC", "QL-QM-ARC", "M-ARC", "Priority-LRU", "LRU", "LFU", "Random"]
    colors = {
        "HA-QM-ARC": "#2ca02c", "QM-ARC": "#1f77b4", "QL-QM-ARC": "#9467bd",
        "M-ARC": "#ff7f0e", "Priority-LRU": "#d62728", "LRU": "#8c564b",
        "LFU": "#e377c2", "Random": "#7f7f7f"
    }
    props_2tier = [0.005, 0.01, 0.05, 0.10]
    for algo in algos:
        means = []
        for p in props_2tier:
            vals = [r["ghr"] for r in raw_results if r["num_tiers"] == 2 and r["cache_proportion"] == p and r["algorithm"] == algo]
            means.append(np.mean(vals) if vals else 0.0)
        ax.plot([p * 100 for p in props_2tier], means, marker="o", label=algo, color=colors.get(algo, None), linewidth=2 if algo == "HA-QM-ARC" else 1.5)

    ax.set_title("Experiment 1: Global Hit Ratio (2-Tier DRAM+SSD, Stable Zipf)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cache Proportion (%)", fontsize=11)
    ax.set_ylabel("Global Hit Ratio (GHR)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "exp1_2tier_ghr.png", dpi=200)
    plt.close()

    # 3-Tier GHR Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    props_3tier = [0.01, 0.05, 0.10]
    for algo in algos:
        means = []
        for p in props_3tier:
            vals = [r["ghr"] for r in raw_results if r["num_tiers"] == 3 and r["cache_proportion"] == p and r["algorithm"] == algo]
            means.append(np.mean(vals) if vals else 0.0)
        ax.plot([p * 100 for p in props_3tier], means, marker="s", label=algo, color=colors.get(algo, None), linewidth=2 if algo == "HA-QM-ARC" else 1.5)

    ax.set_title("Experiment 1: Global Hit Ratio (3-Tier DRAM+SSD+NVMe, Stable Zipf)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cache Proportion (%)", fontsize=11)
    ax.set_ylabel("Global Hit Ratio (GHR)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "exp1_3tier_ghr.png", dpi=200)
    plt.close()


# ============================================================================
# Experiment 2: Sequential Scan Pollution
# ============================================================================

def run_experiment_2_sequential_scan() -> Dict[str, Any]:
    """Execute Experiment 2: Sequential scan pollution evaluation."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT 2: SEQUENTIAL SCAN POLLUTION")
    print("=" * 80)

    exp_dir = PHASE_6_RESULTS_DIR / "experiment_2_sequential_scan"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    algos = ["HA-QM-ARC", "QM-ARC", "QL-QM-ARC", "M-ARC", "Priority-LRU", "LRU"]
    raw_results = []

    for seed in SEEDS:
        trace_path = exp_dir / f"scan_trace_1200reqs_seed{seed}.csv"
        create_scan_pollution_trace(
            output_path=trace_path, num_hot_items=50, scan_unique_items=400,
            phase_len=400, seed=seed,
        )

        for algo in algos:
            # Enable sequential scan detection for HA-QM-ARC
            params = {"enable_scan_detection": True} if algo == "HA-QM-ARC" else None
            cfg = ExperimentConfig(
                algorithm=algo,
                workload=str(trace_path),
                catalog_size=1400,
                cache_proportion=0.05,  # 70 slots (plenty for 50 hot items)
                num_tiers=2,
                num_requests=1200,
                seed=seed,
                output_dir=exp_dir,
                logs_enabled=True,
                algorithm_params=params,
            )
            res = ExperimentRunner.run_experiment(cfg)
            log_path = exp_dir / "logs" / f"{algo}_{seed}.log"
            phase_stats = parse_phase_metrics_from_log(log_path, phase_boundaries=[400, 800, 1200])

            p1_ghr = phase_stats[0]["ghr"] if len(phase_stats) > 0 else 0.0
            p2_ghr = phase_stats[1]["ghr"] if len(phase_stats) > 1 else 0.0
            p3_ghr = phase_stats[2]["ghr"] if len(phase_stats) > 2 else 0.0
            recovery_ratio = (p3_ghr / p1_ghr) if p1_ghr > 0 else 0.0

            row = {
                "experiment": "Exp2_Scan_Pollution",
                "seed": seed,
                "algorithm": algo,
                "overall_ghr": res.metrics.global_hit_ratio,
                "phase1_pre_scan_ghr": p1_ghr,
                "phase2_during_scan_ghr": p2_ghr,
                "phase3_post_scan_ghr": p3_ghr,
                "recovery_ratio": recovery_ratio,
                "total_penalty": res.metrics.total_penalty,
                "penalty_hp": res.metrics.penalty_hp,
                "penalty_lp": res.metrics.penalty_lp,
            }
            raw_results.append(row)

    # Save summary CSV
    csv_path = exp_dir / "aggregated_summary.csv"
    if raw_results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()))
            writer.writeheader()
            writer.writerows(raw_results)

    # Statistical comparison: HA-QM-ARC vs QM-ARC
    ha_p3 = [r["phase3_post_scan_ghr"] for r in raw_results if r["algorithm"] == "HA-QM-ARC"]
    qm_p3 = [r["phase3_post_scan_ghr"] for r in raw_results if r["algorithm"] == "QM-ARC"]
    ha_pen = [r["total_penalty"] for r in raw_results if r["algorithm"] == "HA-QM-ARC"]
    qm_pen = [r["total_penalty"] for r in raw_results if r["algorithm"] == "QM-ARC"]

    st_p3 = compute_paired_statistics(ha_p3, qm_p3)
    st_pen = compute_paired_statistics(ha_pen, qm_pen)

    stats_rows = [
        {
            "metric": "Post-Scan Hit Ratio (Phase 3)",
            "mean_ha": st_p3["mean_a"], "mean_qm": st_p3["mean_b"], "diff_mean": st_p3["mean_diff"],
            "ci_95": f"[{st_p3['ci_lower']:.4f}, {st_p3['ci_upper']:.4f}]", "t_stat": st_p3["t_stat"],
            "p_value": st_p3["p_value"], "cohens_d": st_p3["cohens_d"],
        },
        {
            "metric": "Total SLA Penalty",
            "mean_ha": st_pen["mean_a"], "mean_qm": st_pen["mean_b"], "diff_mean": st_pen["mean_diff"],
            "ci_95": f"[{st_pen['ci_lower']:.2f}, {st_pen['ci_upper']:.2f}]", "t_stat": st_pen["t_stat"],
            "p_value": st_pen["p_value"], "cohens_d": st_pen["cohens_d"],
        },
    ]

    with open(exp_dir / "statistical_analysis.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    # Plot Experiment 2
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(algos))
    width = 0.25

    p1_means = [np.mean([r["phase1_pre_scan_ghr"] for r in raw_results if r["algorithm"] == a]) for a in algos]
    p2_means = [np.mean([r["phase2_during_scan_ghr"] for r in raw_results if r["algorithm"] == a]) for a in algos]
    p3_means = [np.mean([r["phase3_post_scan_ghr"] for r in raw_results if r["algorithm"] == a]) for a in algos]

    ax.bar(x - width, p1_means, width, label="Pre-Scan (Phase 1)", color="#1f77b4")
    ax.bar(x, p2_means, width, label="During Scan (Phase 2)", color="#ff7f0e")
    ax.bar(x + width, p3_means, width, label="Post-Scan Recovery (Phase 3)", color="#2ca02c")

    ax.set_title("Experiment 2: Hot-Set Hit Ratio Under Sequential Scan Pollution", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, fontsize=10)
    ax.set_ylabel("Phase Hit Ratio", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "exp2_scan_pollution_phases.png", dpi=200)
    plt.close()

    print(f"Experiment 2 complete: {len(raw_results)} runs executed.")
    return {"raw_results": raw_results, "stats": stats_rows}


# ============================================================================
# Experiment 3: Sudden Workload Phase Change
# ============================================================================

def run_experiment_3_phase_change() -> Dict[str, Any]:
    """Execute Experiment 3: Non-stationary workload phase shift adaptation."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT 3: WORKLOAD PHASE CHANGE")
    print("=" * 80)

    exp_dir = PHASE_6_RESULTS_DIR / "experiment_3_phase_change"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    algos = ["HA-QM-ARC", "QM-ARC", "QL-QM-ARC", "M-ARC", "LRU"]
    raw_results = []

    for seed in SEEDS:
        trace_path = exp_dir / f"phase_change_1500reqs_seed{seed}.csv"
        create_phase_change_trace(output_path=trace_path, phase_len=500, seed=seed)

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
            phase_stats = parse_phase_metrics_from_log(log_path, phase_boundaries=[500, 1000, 1500])

            pa_ghr = phase_stats[0]["ghr"] if len(phase_stats) > 0 else 0.0
            pb_ghr = phase_stats[1]["ghr"] if len(phase_stats) > 1 else 0.0
            pc_ghr = phase_stats[2]["ghr"] if len(phase_stats) > 2 else 0.0

            row = {
                "experiment": "Exp3_Phase_Change",
                "seed": seed,
                "algorithm": algo,
                "overall_ghr": res.metrics.global_hit_ratio,
                "phaseA_ghr": pa_ghr,
                "phaseB_ghr": pb_ghr,
                "phaseC_ghr": pc_ghr,
                "total_penalty": res.metrics.total_penalty,
                "penalty_hp": res.metrics.penalty_hp,
                "penalty_lp": res.metrics.penalty_lp,
            }
            raw_results.append(row)

    # Save summary CSV
    csv_path = exp_dir / "aggregated_summary.csv"
    if raw_results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()))
            writer.writeheader()
            writer.writerows(raw_results)

    # Statistical comparison: HA-QM-ARC vs QM-ARC
    ha_ghr = [r["overall_ghr"] for r in raw_results if r["algorithm"] == "HA-QM-ARC"]
    qm_ghr = [r["overall_ghr"] for r in raw_results if r["algorithm"] == "QM-ARC"]
    st_ghr = compute_paired_statistics(ha_ghr, qm_ghr)

    stats_rows = [{
        "metric": "Overall Hit Ratio (Phases A-C)",
        "mean_ha": st_ghr["mean_a"], "mean_qm": st_ghr["mean_b"], "diff_mean": st_ghr["mean_diff"],
        "ci_95": f"[{st_ghr['ci_lower']:.4f}, {st_ghr['ci_upper']:.4f}]", "t_stat": st_ghr["t_stat"],
        "p_value": st_ghr["p_value"], "cohens_d": st_ghr["cohens_d"],
    }]

    with open(exp_dir / "statistical_analysis.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    # Plot Experiment 3
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(algos))
    width = 0.25
    pa_m = [np.mean([r["phaseA_ghr"] for r in raw_results if r["algorithm"] == a]) for a in algos]
    pb_m = [np.mean([r["phaseB_ghr"] for r in raw_results if r["algorithm"] == a]) for a in algos]
    pc_m = [np.mean([r["phaseC_ghr"] for r in raw_results if r["algorithm"] == a]) for a in algos]

    ax.bar(x - width, pa_m, width, label="Phase A (Initial Set)", color="#1f77b4")
    ax.bar(x, pb_m, width, label="Phase B (Shifted Set)", color="#ff7f0e")
    ax.bar(x + width, pc_m, width, label="Phase C (Return Set)", color="#2ca02c")

    ax.set_title("Experiment 3: Hit Ratio Across Popularity Phase Shifts", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, fontsize=10)
    ax.set_ylabel("Phase Hit Ratio", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "exp3_phase_change_adaptation.png", dpi=200)
    plt.close()

    print(f"Experiment 3 complete: {len(raw_results)} runs executed.")
    return {"raw_results": raw_results, "stats": stats_rows}


# ============================================================================
# Experiment 4: High-Priority Traffic Surge (QoS Stress)
# ============================================================================

def run_experiment_4_qos_surge() -> Dict[str, Any]:
    """Execute Experiment 4: QoS priority stress and dynamic lambda attenuation."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT 4: HIGH-PRIORITY TRAFFIC SURGE")
    print("=" * 80)

    exp_dir = PHASE_6_RESULTS_DIR / "experiment_4_qos_surge"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    algos = ["HA-QM-ARC", "QM-ARC", "QL-QM-ARC", "Priority-LRU", "LRU"]
    raw_results = []

    for seed in SEEDS:
        trace_path = exp_dir / f"qos_surge_1500reqs_seed{seed}.csv"
        create_qos_surge_trace(output_path=trace_path, phase_len=500, seed=seed)

        for algo in algos:
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
            )
            res = ExperimentRunner.run_experiment(cfg)
            row = {
                "experiment": "Exp4_QoS_Surge",
                "seed": seed,
                "algorithm": algo,
                "ghr": res.metrics.global_hit_ratio,
                "hphr": res.metrics.high_priority_hit_ratio,
                "lphr": res.metrics.low_priority_hit_ratio,
                "total_penalty": res.metrics.total_penalty,
                "penalty_hp": res.metrics.penalty_hp,
                "penalty_lp": res.metrics.penalty_lp,
            }
            raw_results.append(row)

    # Save summary CSV
    csv_path = exp_dir / "aggregated_summary.csv"
    if raw_results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()))
            writer.writeheader()
            writer.writerows(raw_results)

    # Statistical comparison: HA-QM-ARC vs QM-ARC on HP Penalty and HP Hit Ratio
    ha_hphr = [r["hphr"] for r in raw_results if r["algorithm"] == "HA-QM-ARC"]
    qm_hphr = [r["hphr"] for r in raw_results if r["algorithm"] == "QM-ARC"]
    ha_pen_hp = [r["penalty_hp"] for r in raw_results if r["algorithm"] == "HA-QM-ARC"]
    qm_pen_hp = [r["penalty_hp"] for r in raw_results if r["algorithm"] == "QM-ARC"]

    st_hphr = compute_paired_statistics(ha_hphr, qm_hphr)
    st_pen_hp = compute_paired_statistics(ha_pen_hp, qm_pen_hp)

    stats_rows = [
        {
            "metric": "High-Priority Hit Ratio",
            "mean_ha": st_hphr["mean_a"], "mean_qm": st_hphr["mean_b"], "diff_mean": st_hphr["mean_diff"],
            "ci_95": f"[{st_hphr['ci_lower']:.4f}, {st_hphr['ci_upper']:.4f}]", "t_stat": st_hphr["t_stat"],
            "p_value": st_hphr["p_value"], "cohens_d": st_hphr["cohens_d"],
        },
        {
            "metric": "High-Priority SLA Penalty",
            "mean_ha": st_pen_hp["mean_a"], "mean_qm": st_pen_hp["mean_b"], "diff_mean": st_pen_hp["mean_diff"],
            "ci_95": f"[{st_pen_hp['ci_lower']:.2f}, {st_pen_hp['ci_upper']:.2f}]", "t_stat": st_pen_hp["t_stat"],
            "p_value": st_pen_hp["p_value"], "cohens_d": st_pen_hp["cohens_d"],
        },
    ]

    with open(exp_dir / "statistical_analysis.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    # Plot Experiment 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    hp_m = [np.mean([r["hphr"] for r in raw_results if r["algorithm"] == a]) for a in algos]
    lp_m = [np.mean([r["lphr"] for r in raw_results if r["algorithm"] == a]) for a in algos]
    pen_hp_m = [np.mean([r["penalty_hp"] for r in raw_results if r["algorithm"] == a]) for a in algos]

    x = np.arange(len(algos))
    ax1.bar(x - 0.15, hp_m, 0.3, label="HP Hit Ratio", color="#2ca02c")
    ax1.bar(x + 0.15, lp_m, 0.3, label="LP Hit Ratio", color="#1f77b4")
    ax1.set_title("Hit Ratio by Priority Under Traffic Surge", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, rotation=20, fontsize=9)
    ax1.set_ylabel("Hit Ratio", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    ax2.bar(x, pen_hp_m, 0.45, color="#d62728")
    ax2.set_title("High-Priority SLA Penalty (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, rotation=20, fontsize=9)
    ax2.set_ylabel("SLA Penalty Units", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "exp4_qos_surge_comparison.png", dpi=200)
    plt.close()

    print(f"Experiment 4 complete: {len(raw_results)} runs executed.")
    return {"raw_results": raw_results, "stats": stats_rows}


# ============================================================================
# Experiment 5: Ablation Study
# ============================================================================

def run_experiment_5_ablation() -> Dict[str, Any]:
    """Execute Experiment 5: Architectural component ablation study."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT 5: ABLATION STUDY")
    print("=" * 80)

    exp_dir = PHASE_6_RESULTS_DIR / "experiment_5_ablation"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "Complete HA-QM-ARC": {"enable_scan_detection": True},
        "w/o Hit Gradient": {"omega_grad": 0.0, "enable_scan_detection": True},
        "w/o Burst Detection": {"omega_burst": 0.0, "enable_scan_detection": True},
        "w/o Sequential Detection": {"enable_scan_detection": False, "omega_scan": 0.0},
        "w/o Dynamic QoS": {"kappa_qos": 0.0, "enable_scan_detection": True},
    }

    raw_results = []

    # Evaluate across 2 representative workloads: Scan Pollution and QoS Surge
    for seed in SEEDS:
        scan_trace = exp_dir / f"ablation_scan_seed{seed}.csv"
        create_scan_pollution_trace(scan_trace, num_hot_items=50, scan_unique_items=400, phase_len=400, seed=seed)

        surge_trace = exp_dir / f"ablation_surge_seed{seed}.csv"
        create_qos_surge_trace(surge_trace, phase_len=500, seed=seed)

        workloads = [("Scan_Pollution", scan_trace, 1200, 0.05, 1400), ("QoS_Surge", surge_trace, 1500, 0.01, 10000)]

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
                    "experiment": "Exp5_Ablation",
                    "workload": wl_name,
                    "variant": v_name,
                    "seed": seed,
                    "ghr": res.metrics.global_hit_ratio,
                    "hphr": res.metrics.high_priority_hit_ratio,
                    "total_penalty": res.metrics.total_penalty,
                    "penalty_hp": res.metrics.penalty_hp,
                }
                raw_results.append(row)

    # Save summary CSV
    csv_path = exp_dir / "aggregated_summary.csv"
    if raw_results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()))
            writer.writeheader()
            writer.writerows(raw_results)

    # Plot Experiment 5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    v_keys = list(variants.keys())
    x = np.arange(len(v_keys))

    scan_ghrs = [np.mean([r["ghr"] for r in raw_results if r["workload"] == "Scan_Pollution" and r["variant"] == v]) for v in v_keys]
    surge_pens = [np.mean([r["penalty_hp"] for r in raw_results if r["workload"] == "QoS_Surge" and r["variant"] == v]) for v in v_keys]

    ax1.bar(x, scan_ghrs, color="#1f77b4", width=0.5)
    ax1.set_title("Scan Pollution Hit Ratio by Ablated Variant", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(v_keys, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Global Hit Ratio", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    ax2.bar(x, surge_pens, color="#d62728", width=0.5)
    ax2.set_title("QoS Surge HP Penalty by Ablated Variant", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(v_keys, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("HP SLA Penalty Units", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "exp5_ablation_summary.png", dpi=200)
    plt.close()

    print(f"Experiment 5 complete: {len(raw_results)} runs executed.")
    return {"raw_results": raw_results}


# ============================================================================
# Experiment 6: Sensitivity Analysis
# ============================================================================

def run_experiment_6_sensitivity() -> Dict[str, Any]:
    """Execute Experiment 6: History window W and controller parameter sensitivity."""
    print("\n" + "=" * 80)
    print("STARTING EXPERIMENT 6: SENSITIVITY ANALYSIS")
    print("=" * 80)

    exp_dir = PHASE_6_RESULTS_DIR / "experiment_6_sensitivity"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    windows = [20, 50, 100, 200, 500]
    raw_results = []

    # 1. Window Size Sensitivity on Stable Zipf
    for W in windows:
        for seed in SEEDS:
            trace_path = resources.SYNTHETIC_DIR / f"synth_items10000_reqs1000_seed{seed}.csv"
            cfg = ExperimentConfig(
                algorithm="HA-QM-ARC",
                workload=str(trace_path),
                catalog_size=10000,
                cache_proportion=0.01,
                num_tiers=2,
                num_requests=1000,
                seed=seed,
                output_dir=exp_dir,
                logs_enabled=False,
                algorithm_params={"window_size": W},
            )
            res = ExperimentRunner.run_experiment(cfg)
            row = {
                "experiment": "Exp6_Window_Sensitivity",
                "param_name": "window_size",
                "param_value": W,
                "seed": seed,
                "ghr": res.metrics.global_hit_ratio,
                "total_penalty": res.metrics.total_penalty,
                "penalty_hp": res.metrics.penalty_hp,
                "penalty_lp": res.metrics.penalty_lp,
                "wall_time": res.metrics.simulation_wall_time_seconds,
            }
            raw_results.append(row)

    # 2. Kappa QoS Sensitivity on Surge Workload
    kappas = [0.10, 0.30, 0.50, 0.70, 0.90]
    for k in kappas:
        for seed in SEEDS:
            surge_trace = exp_dir / f"sens_surge_seed{seed}.csv"
            if not surge_trace.is_file():
                create_qos_surge_trace(surge_trace, phase_len=500, seed=seed)
            cfg = ExperimentConfig(
                algorithm="HA-QM-ARC",
                workload=str(surge_trace),
                catalog_size=10000,
                cache_proportion=0.01,
                num_tiers=2,
                num_requests=1500,
                seed=seed,
                output_dir=exp_dir,
                logs_enabled=False,
                algorithm_params={"kappa_qos": k},
            )
            res = ExperimentRunner.run_experiment(cfg)
            row = {
                "experiment": "Exp6_Kappa_Sensitivity",
                "param_name": "kappa_qos",
                "param_value": k,
                "seed": seed,
                "ghr": res.metrics.global_hit_ratio,
                "total_penalty": res.metrics.total_penalty,
                "penalty_hp": res.metrics.penalty_hp,
                "penalty_lp": res.metrics.penalty_lp,
                "wall_time": res.metrics.simulation_wall_time_seconds,
            }
            raw_results.append(row)

    # Save summary CSV
    csv_path = exp_dir / "aggregated_summary.csv"
    fieldnames = [
        "experiment", "param_name", "param_value", "seed",
        "ghr", "total_penalty", "penalty_hp", "penalty_lp", "wall_time"
    ]
    if raw_results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_results)

    # Plot Experiment 6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    w_means = [np.mean([r["ghr"] for r in raw_results if r["param_name"] == "window_size" and r["param_value"] == w]) for w in windows]
    ax1.plot(windows, w_means, marker="o", color="#2ca02c", linewidth=2)
    ax1.set_title("Hit Ratio vs History Window Size (W)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("History Window Capacity W (requests)", fontsize=10)
    ax1.set_ylabel("Global Hit Ratio", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    k_hp_pens = [np.mean([r["penalty_hp"] for r in raw_results if r["param_name"] == "kappa_qos" and r["param_value"] == k]) for k in kappas]
    ax2.plot(kappas, k_hp_pens, marker="s", color="#d62728", linewidth=2)
    ax2.set_title("High-Priority Penalty vs Kappa QoS Weight", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Kappa QoS Parameter", fontsize=10)
    ax2.set_ylabel("HP SLA Penalty Units", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(plots_dir / "exp6_sensitivity_curves.png", dpi=200)
    plt.close()

    print(f"Experiment 6 complete: {len(raw_results)} runs executed.")
    return {"raw_results": raw_results}


# ============================================================================
# Main Execution Orchestrator
# ============================================================================

def main():
    t_start = time.time()
    print("================================================================================")
    print("LAUNCHING PHASE 6 COMPREHENSIVE EXPERIMENTAL BENCHMARKING SUITE")
    print(f"Results Directory: {PHASE_6_RESULTS_DIR}")
    print(f"Random Seeds: {SEEDS}")
    print("================================================================================")

    # Resume-safe execution for all 6 experiments
    if not (PHASE_6_RESULTS_DIR / "experiment_1_stable_zipf" / "aggregated_summary.csv").is_file():
        exp1_res = run_experiment_1_stable_zipf()
    else:
        print("Experiment 1 already completed. Skipping.")

    if not (PHASE_6_RESULTS_DIR / "experiment_2_sequential_scan" / "aggregated_summary.csv").is_file():
        exp2_res = run_experiment_2_sequential_scan()
    else:
        print("Experiment 2 already completed. Skipping.")

    if not (PHASE_6_RESULTS_DIR / "experiment_3_phase_change" / "aggregated_summary.csv").is_file():
        exp3_res = run_experiment_3_phase_change()
    else:
        print("Experiment 3 already completed. Skipping.")

    if not (PHASE_6_RESULTS_DIR / "experiment_4_qos_surge" / "aggregated_summary.csv").is_file():
        exp4_res = run_experiment_4_qos_surge()
    else:
        print("Experiment 4 already completed. Skipping.")

    if not (PHASE_6_RESULTS_DIR / "experiment_5_ablation" / "aggregated_summary.csv").is_file():
        exp5_res = run_experiment_5_ablation()
    else:
        print("Experiment 5 already completed. Skipping.")

    exp6_res = run_experiment_6_sensitivity()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL PHASE 6 EXPERIMENTS COMPLETED IN {total_time:.2f} SECONDS (~{total_time/60.0:.2f} MINUTES)!")
    print("================================================================================")


if __name__ == "__main__":
    main()
