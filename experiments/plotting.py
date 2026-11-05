import json
from pathlib import Path
from typing import List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CLI and test environments
import matplotlib.pyplot as plt
import pandas as pd


def load_summary_df(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate an experiments summary CSV file."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Summary CSV file is empty: {path}")
    return df


def plot_hit_ratios(
    summary_csv_path: Union[str, Path],
    output_image_path: Optional[Union[str, Path]] = None,
    title: str = "Cache Hit Ratio Comparison",
) -> Path:
    """Generate a grouped bar chart comparing Global, High-Priority, and Low-Priority hit ratios.

    Args:
        summary_csv_path: Path to experiments_summary.csv.
        output_image_path: Optional output path for the saved PNG figure.
        title: Chart title.

    Returns:
        Path to the saved figure.
    """
    df = load_summary_df(summary_csv_path)
    target_path = Path(output_image_path) if output_image_path else Path(summary_csv_path).parent / "hit_ratios.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # If multiple runs exist for same algorithm, group by algorithm and take the latest or mean
    metrics_cols = ["global_hit_ratio", "high_priority_hit_ratio", "low_priority_hit_ratio"]
    grouped = df.groupby("algorithm")[metrics_cols].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(grouped))
    width = 0.25

    ax.bar([i - width for i in x], grouped["global_hit_ratio"], width=width, label="Global Hit Ratio", color="#2b5c8f")
    ax.bar(x, grouped["high_priority_hit_ratio"], width=width, label="High-Priority Hit Ratio", color="#2ca02c")
    ax.bar([i + width for i in x], grouped["low_priority_hit_ratio"], width=width, label="Low-Priority Hit Ratio", color="#d62728")

    ax.set_ylabel("Hit Ratio (0.0 - 1.0)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped.index, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close(fig)
    return target_path


def plot_penalties(
    summary_csv_path: Union[str, Path],
    output_image_path: Optional[Union[str, Path]] = None,
    title: str = "SLA Cache Miss Penalty Comparison",
) -> Path:
    """Generate a stacked bar chart of High-Priority vs Low-Priority SLA miss penalties."""
    df = load_summary_df(summary_csv_path)
    target_path = Path(output_image_path) if output_image_path else Path(summary_csv_path).parent / "penalties.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("algorithm")[["penalty_hp", "penalty_lp"]].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(grouped))
    width = 0.45

    ax.bar(x, grouped["penalty_hp"], width=width, label="High-Priority Penalty", color="#ff7f0e")
    ax.bar(x, grouped["penalty_lp"], width=width, bottom=grouped["penalty_hp"], label="Low-Priority Penalty", color="#1f77b4")

    ax.set_ylabel("Accumulated Penalty")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped.index, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close(fig)
    return target_path


def plot_retrieval_latencies(
    summary_csv_path: Union[str, Path],
    output_image_path: Optional[Union[str, Path]] = None,
    title: str = "Average Content Retrieval Latency",
) -> Path:
    """Generate a bar chart comparing Average Retrieval Latencies (Global, HP, LP)."""
    df = load_summary_df(summary_csv_path)
    target_path = Path(output_image_path) if output_image_path else Path(summary_csv_path).parent / "retrieval_latencies.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ["avg_retrieval_latency", "avg_retrieval_latency_hp", "avg_retrieval_latency_lp"]
    # Convert seconds to milliseconds for chart readability
    grouped = df.groupby("algorithm")[cols].mean() * 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(grouped))
    width = 0.25

    ax.bar([i - width for i in x], grouped["avg_retrieval_latency"], width=width, label="Avg Latency (Global)", color="#9467bd")
    ax.bar(x, grouped["avg_retrieval_latency_hp"], width=width, label="Avg Latency (HP)", color="#2ca02c")
    ax.bar([i + width for i in x], grouped["avg_retrieval_latency_lp"], width=width, label="Avg Latency (LP)", color="#d62728")

    ax.set_ylabel("Latency (ms)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped.index, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close(fig)
    return target_path


def plot_tier_breakdown_from_json(
    json_path: Union[str, Path],
    output_image_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Plot per-tier occupancy and hit ratio from an experiment result JSON."""
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Result JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_path = Path(output_image_path) if output_image_path else path.parent / f"{path.stem}_tiers.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    tier_data = data.get("metrics", {}).get("tier_breakdown", [])
    if not tier_data:
        raise ValueError("No tier breakdown found in result JSON.")

    tier_names = [t["tier_name"] for t in tier_data]
    hit_ratios = [t["hit_ratio"] for t in tier_data]
    occupancies = [t["occupancy_ratio"] for t in tier_data]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = range(len(tier_names))
    width = 0.35

    ax1.bar([i - width / 2 for i in x], hit_ratios, width=width, label="Tier Hit Ratio", color="#1f77b4")
    ax1.bar([i + width / 2 for i in x], occupancies, width=width, label="Occupancy Ratio", color="#ff7f0e")

    ax1.set_ylabel("Ratio (0.0 - 1.0)")
    ax1.set_title(f"Tier Performance - {data.get('algorithm')} ({data.get('experiment_id')})", fontsize=12)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(tier_names)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close(fig)
    return target_path
