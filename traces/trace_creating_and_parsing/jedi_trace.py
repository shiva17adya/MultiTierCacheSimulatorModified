import argparse
import csv
from pathlib import Path
import random
import sys
import time
from typing import Optional, Dict

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import resources

# Jedi CDN Video trace characteristics based on Akamai model (IMC '22 / QM-ARC paper):
# - 40,000 unique video objects (chunks)
# - Long-tail popularity (modeled with generalized Pareto / bounded Zipf with heavy tail)
# - Request rate: ~10 requests/s
# - 20% high priority, 80% low priority content
# - Default NDN packet format: 'data_back', 'timestamp', 'name', 'size', 'priority', 'InterestLifetime', 'response_time'


def parse_raw_jedi_trace(
    raw_input_path: Path,
    output_path: Optional[Path] = None,
    high_priority_ratio: float = 0.2,
    trace_len_limit: int = 100000,
    seed: Optional[int] = 42,
) -> Path:
    """Parse an authentic raw Jedi trace file (e.g. jedi_w / jedi_eu).

    Raw file format: comma-delimited `timestamp, object_id, total_object_size` (size in KB or bytes).
    """
    raw_input_path = Path(raw_input_path)
    if not raw_input_path.is_file():
        raise FileNotFoundError(f"Raw Jedi trace not found: {raw_input_path}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if output_path is None:
        resources.JEDI_DIR.mkdir(parents=True, exist_ok=True)
        output_path = resources.JEDI_DIR / f"jedi_video_from_raw_{raw_input_path.stem}.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[AUTHENTIC PARSER] Parsing raw Jedi file: {raw_input_path} -> {output_path}")
    object_priority: Dict[str, str] = {}
    object_response_time: Dict[str, float] = {}
    last_timestamp = 0.0
    processed_count = 0

    with open(raw_input_path, "r", encoding="utf-8", errors="ignore") as in_f, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as out_f:
        writer = csv.writer(out_f)
        for line in in_f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            parts = [p.strip() for p in line_str.split(",")]
            if len(parts) < 3:
                continue
            try:
                raw_ts = float(parts[0])
                obj_id = str(parts[1])
                obj_size = int(float(parts[2]))
                # Normalize timestamp to seconds
                timestamp = raw_ts / 1000.0 if raw_ts > 100000 else raw_ts

                if obj_id not in object_priority:
                    object_priority[obj_id] = "h" if random.random() < high_priority_ratio else "l"
                    object_response_time[obj_id] = float(np.random.uniform(0.01, 0.20))

                # Maintain monotonic request times
                if last_timestamp >= timestamp:
                    timestamp = last_timestamp + float(np.random.uniform(0.01, 0.10))

                writer.writerow([
                    "d",
                    f"{timestamp:.6f}",
                    obj_id,
                    obj_size,
                    object_priority[obj_id],
                    4.0,  # Interest lifetime (s)
                    f"{object_response_time[obj_id]:.6f}",
                ])
                last_timestamp = timestamp
                processed_count += 1
                if 0 < trace_len_limit <= processed_count:
                    break
            except (ValueError, IndexError):
                continue

    print(f"[AUTHENTIC PARSER] Processed {processed_count} requests into {output_path}")
    return output_path


def generate_synthetic_jedi_workload(
    n_unique_items: int = 40000,
    num_requests: int = 100000,
    arrival_rate: float = 10.0,
    high_priority_ratio: float = 0.2,
    tail_shape: float = 0.8,
    min_chunk_size: int = 1024,
    max_chunk_size: int = 8192,
    min_rtt: float = 0.01,
    max_rtt: float = 0.20,
    seed: Optional[int] = 42,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a synthetic video workload that approximates Akamai CDN / Jedi characteristics.

    DISCLAIMER: This is a SYNTHETIC APPROXIMATION modeled on published Jedi trace parameters
    (40,000 objects, heavy-tailed access frequency, 10 req/s, 20% high priority). It is NOT
    the raw proprietary Akamai dataset.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if output_path is None:
        resources.JEDI_DIR.mkdir(parents=True, exist_ok=True)
        filename = (
            f"jedi_video_synthetic_approx_items{n_unique_items}_hp{high_priority_ratio}_"
            f"reqs{num_requests}.csv"
        )
        output_path = resources.JEDI_DIR / filename
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SYNTHETIC JEDI APPROXIMATION] Generating workload -> {output_path}")
    print(f"  Note: This is a synthetic model mimicking Jedi CDN video access patterns.")
    print(f"  Unique objects: {n_unique_items}, Requests: {num_requests}, Rate: {arrival_rate} req/s")

    # Generate catalog with video chunk properties
    catalog: Dict[int, Dict] = {}
    for i in range(n_unique_items):
        size = int(round(np.random.uniform(min_chunk_size, max_chunk_size)))
        rtt = float(np.random.uniform(min_rtt, max_rtt))
        catalog[i] = {
            "name": f"video_chunk_{i}",
            "size": size,
            "response_time": rtt,
        }

    # Precompute heavy-tail distribution (Bounded Pareto / Zipf tail)
    # Jedi exhibits longer tail than standard alpha=1.0 Zipf
    ranks = np.arange(1, n_unique_items + 1)
    probs = 1.0 / (ranks ** tail_shape)
    probs /= probs.sum()

    # Pre-sample request object indices and priorities
    selected_indices = np.random.choice(n_unique_items, size=num_requests, p=probs)
    hp_flags = np.random.random(num_requests) < high_priority_ratio

    current_time = 0.0
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        for req_idx in range(num_requests):
            inter_arrival = random.expovariate(arrival_rate)
            current_time += inter_arrival
            item = catalog[selected_indices[req_idx]]
            priority = "h" if hp_flags[req_idx] else "l"
            writer.writerow([
                "d",
                f"{current_time:.6f}",
                item["name"],
                item["size"],
                priority,
                4.0,  # Interest lifetime
                f"{item['response_time']:.6f}",
            ])

    print(f"[SYNTHETIC JEDI APPROXIMATION] Completed {num_requests} requests -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jedi Video Workload Processor / Generator")
    parser.add_argument("--raw-input", type=str, default=None, help="Path to authentic raw Jedi trace file (if available)")
    parser.add_argument("--output", type=str, default=None, help="Destination output CSV path")
    parser.add_argument("--items", type=int, default=40000, help="Unique objects for synthetic generation")
    parser.add_argument("--requests", type=int, default=10000, help="Request count")
    parser.add_argument("--rate", type=float, default=10.0, help="Arrival rate (req/s)")
    parser.add_argument("--priority-ratio", type=float, default=0.20, help="High priority proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    default_raw = resources.RAW_DATASET_DIR / "jedi_w"
    if args.raw_input:
        parse_raw_jedi_trace(
            raw_input_path=Path(args.raw_input),
            output_path=Path(args.output) if args.output else None,
            high_priority_ratio=args.priority_ratio,
            trace_len_limit=args.requests,
            seed=args.seed,
        )
    elif default_raw.is_file():
        parse_raw_jedi_trace(
            raw_input_path=default_raw,
            output_path=Path(args.output) if args.output else None,
            high_priority_ratio=args.priority_ratio,
            trace_len_limit=args.requests,
            seed=args.seed,
        )
    else:
        print(f"Authentic raw Jedi trace '{default_raw}' not found.")
        print("Falling back to generating synthetic Jedi-approximated video workload.")
        generate_synthetic_jedi_workload(
            n_unique_items=args.items,
            num_requests=args.requests,
            arrival_rate=args.rate,
            high_priority_ratio=args.priority_ratio,
            seed=args.seed,
            output_path=Path(args.output) if args.output else None,
        )
