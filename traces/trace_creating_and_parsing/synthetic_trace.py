import argparse
import csv
from pathlib import Path
import random
import sys
import time
from typing import Optional, Dict

# Ensure project root is in sys.path regardless of execution directory
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from common.packet import Packet
from common.zipf import zipf_distribution
import resources

# Time is in seconds, size is in bytes
# NDN trace column names: 'data_back', 'timestamp', 'name', 'size', 'priority', 'InterestLifetime', 'response_time'


def generate_synthetic_trace(
    n_unique_items: int = 10000,
    zipf_alpha: float = 1.0,
    high_priority_ratio: float = 0.2,
    num_requests: int = 100000,
    poisson_lambda: float = 10.0,
    min_data_size: int = 100,
    max_data_size: int = 8192,
    min_data_rtt: float = 0.01,
    max_data_rtt: float = 0.20,
    interest_lifetime: float = 4.0,
    seed: Optional[int] = 42,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a reproducible synthetic NDN trace following a Zipf popularity distribution.

    Args:
        n_unique_items: Number of distinct objects in the catalog.
        zipf_alpha: Exponent alpha of the Zipf distribution (default 1.0).
        high_priority_ratio: Proportion of requests belonging to high priority (default 0.20).
        num_requests: Total number of interest packets to generate.
        poisson_lambda: Average arrival rate in requests per second (default 10.0).
        min_data_size: Minimum object size in bytes (default 100).
        max_data_size: Maximum object size in bytes (default 8192 = 8 KB).
        min_data_rtt: Minimum round trip / retrieval time in seconds (default 0.01s).
        max_data_rtt: Maximum round trip / retrieval time in seconds (default 0.20s).
        interest_lifetime: NDN Interest lifetime in seconds (default 4.0s).
        seed: Random seed for full reproducibility.
        output_path: Destination CSV path. If None, saves in resources/dataset_synthetic/.

    Returns:
        Path to the generated CSV file.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start_time = time.time()

    if output_path is None:
        resources.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
        filename = (
            f"synthetic-items{n_unique_items}_alpha{zipf_alpha}_hp{high_priority_ratio}_"
            f"reqs{num_requests}_lambda{int(poisson_lambda)}.csv"
        )
        output_path = resources.SYNTHETIC_DIR / filename
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic trace -> {output_path}")
    print(f"  Unique objects: {n_unique_items}, Zipf alpha: {zipf_alpha}")
    print(f"  Requests: {num_requests}, Arrival rate: {poisson_lambda} req/s")
    print(f"  Priority target: {high_priority_ratio * 100}% high, {(1 - high_priority_ratio) * 100}% low")

    # Catalog generation: assign each unique item an intrinsic size and base RTT
    catalog: Dict[int, Dict] = {}
    for i in range(n_unique_items):
        size = int(round(np.random.uniform(min_data_size, max_data_size)))
        base_rtt = float(np.random.uniform(min_data_rtt, max_data_rtt))
        catalog[i] = {
            "name": str(i),
            "size": size,
            "response_time": base_rtt,
        }

    # Pre-draw priority per request to precisely match target ratio
    high_priority_flags = np.random.random(num_requests) < high_priority_ratio

    current_time = 0.0
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for req_idx in range(num_requests):
            # Poisson inter-arrival time: delta_t ~ Exponential(lambda)
            inter_arrival = random.expovariate(poisson_lambda)
            current_time += inter_arrival

            # Zipfian item selection (0-indexed)
            item_rank = zipf_distribution(zipf_alpha, n_unique_items) - 1
            item = catalog[item_rank]
            priority = "h" if high_priority_flags[req_idx] else "l"

            writer.writerow([
                "d",
                f"{current_time:.6f}",
                item["name"],
                item["size"],
                priority,
                interest_lifetime,
                f"{item['response_time']:.6f}",
            ])

    elapsed = time.time() - start_time
    print(f"Successfully generated {num_requests} requests in {elapsed:.2f} seconds.")
    return output_path


class TraceCreator:
    """Legacy compatibility class preserving the original interface."""

    def __init__(
        self,
        n_unique_items: int,
        high_priority_content_percentage: float,
        pareto_alpha: float,
        zipf_alpha: float,
        poisson_lambda: float,
        min_data_size: int,
        max_data_size: int,
        min_data_rtt: float,
        max_data_rtt: float,
        interest_lifetime: float,
        traffic_period: int,
        seed: Optional[int] = 42,
        num_requests: Optional[int] = None,
    ):
        target_requests = (
            num_requests if num_requests is not None else int(poisson_lambda * traffic_period * 60)
        )
        self.output_file = generate_synthetic_trace(
            n_unique_items=n_unique_items,
            zipf_alpha=zipf_alpha,
            high_priority_ratio=high_priority_content_percentage,
            num_requests=target_requests,
            poisson_lambda=poisson_lambda,
            min_data_size=min_data_size,
            max_data_size=max_data_size,
            min_data_rtt=min_data_rtt,
            max_data_rtt=max_data_rtt,
            interest_lifetime=interest_lifetime,
            seed=seed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Zipf-like NDN Trace Generator")
    parser.add_argument("--items", type=int, default=10000, help="Number of unique objects")
    parser.add_argument("--alpha", type=float, default=1.0, help="Zipf distribution alpha parameter")
    parser.add_argument("--priority-ratio", type=float, default=0.20, help="High priority request ratio")
    parser.add_argument("--requests", type=int, default=10000, help="Total number of requests")
    parser.add_argument("--rate", type=float, default=10.0, help="Poisson arrival rate (req/s)")
    parser.add_argument("--min-size", type=int, default=100, help="Min data size in bytes")
    parser.add_argument("--max-size", type=int, default=8192, help="Max data size in bytes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output file path")

    args = parser.parse_args()
    generate_synthetic_trace(
        n_unique_items=args.items,
        zipf_alpha=args.alpha,
        high_priority_ratio=args.priority_ratio,
        num_requests=args.requests,
        poisson_lambda=args.rate,
        min_data_size=args.min_size,
        max_data_size=args.max_size,
        seed=args.seed,
        output_path=Path(args.output) if args.output else None,
    )
