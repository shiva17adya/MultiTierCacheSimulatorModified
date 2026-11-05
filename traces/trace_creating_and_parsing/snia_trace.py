import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Optional, Dict

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import resources

# IBM Object Store Trace Parser (SNIA IOTTA Trace 36308 / 001 Part 0)
# Reference: http://iotta.snia.org/traces/key-value/36305?only=36308
# Expected raw input format:
#   timestamp(ms) request_type object_id total_object_size start_offset end_offset
# Filter: Only REST.GET.OBJECT requests are processed.


def parse_ibm_snia_trace(
    raw_input_path: Path,
    output_path: Optional[Path] = None,
    high_priority_ratio: float = 0.2,
    trace_len_limit: int = 100000,
    seed: Optional[int] = 42,
) -> Path:
    """Parse a SNIA IOTTA IBM Object Store raw trace file into NDN simulator CSV format.

    Args:
        raw_input_path: Path to raw input file.
        output_path: Path to write the parsed CSV. If None, saves in resources/dataset_snia/.
        high_priority_ratio: Fraction of objects assigned to high-priority (default 0.20).
        trace_len_limit: Maximum number of GET requests to extract (-1 for unlimited).
        seed: Random seed for priority and response time assignment.

    Returns:
        Path to the output CSV file.
    """
    raw_input_path = Path(raw_input_path)
    if not raw_input_path.is_file():
        raise FileNotFoundError(
            f"Raw IBM trace file not found: {raw_input_path}\n"
            f"Please download 'IBMObjectStoreTrace001Part0' from the SNIA IOTTA repository:\n"
            f"  URL: http://iotta.snia.org/traces/key-value/36305?only=36308\n"
            f"and place it at: {resources.RAW_DATASET_DIR / 'IBMObjectStoreTrace001Part0'}"
        )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if output_path is None:
        resources.SNIA_DIR.mkdir(parents=True, exist_ok=True)
        out_name = f"ibm_object_store_parsed_{raw_input_path.stem}.csv"
        output_path = resources.SNIA_DIR / out_name
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[IBM SNIA PARSER] Parsing raw trace: {raw_input_path} -> {output_path}")

    object_priority: Dict[str, str] = {}
    object_response_time: Dict[str, float] = {}
    get_request_count = 0
    total_lines = 0

    with open(raw_input_path, "r", encoding="utf-8", errors="ignore") as in_f, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as out_f:
        writer = csv.writer(out_f)
        for line in in_f:
            total_lines += 1
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            # Split on whitespace
            parts = line_str.split()
            if len(parts) < 4:
                continue

            try:
                raw_timestamp = int(parts[0])
                request_type = parts[1]
                object_id = parts[2]
                total_object_size = int(parts[3])

                # Convert millisecond timestamp to seconds
                timestamp = raw_timestamp / 1000.0

                # Filter strictly for REST.GET.OBJECT requests per paper methodology
                if request_type == "REST.GET.OBJECT":
                    if object_id not in object_priority:
                        object_priority[object_id] = (
                            "h" if random.random() < high_priority_ratio else "l"
                        )
                        # Paper response time range (0.01s - 0.20s = 10ms - 200ms)
                        object_response_time[object_id] = float(np.random.uniform(0.01, 0.20))

                    interest_lifetime = 4.0  # 4 seconds
                    writer.writerow([
                        "d",
                        f"{timestamp:.6f}",
                        object_id,
                        total_object_size,
                        object_priority[object_id],
                        interest_lifetime,
                        f"{object_response_time[object_id]:.6f}",
                    ])
                    get_request_count += 1
                    if 0 < trace_len_limit <= get_request_count:
                        break
            except (ValueError, IndexError):
                continue

    print(
        f"[IBM SNIA PARSER] Finished! Processed {total_lines} raw lines, "
        f"extracted {get_request_count} REST.GET.OBJECT requests."
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNIA IBM Object Store Trace Parser")
    parser.add_argument("--raw-input", type=str, default=None, help="Path to raw IBMObjectStoreTrace file")
    parser.add_argument("--output", type=str, default=None, help="Destination output CSV file")
    parser.add_argument("--limit", type=int, default=100000, help="Max GET requests to extract")
    parser.add_argument("--priority-ratio", type=float, default=0.20, help="High priority proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    default_raw = resources.RAW_DATASET_DIR / "IBMObjectStoreTrace001Part0"
    sample_raw = resources.RAW_DATASET_DIR / "sample_IBMObjectStoreTrace001Part0.txt"

    target_input = None
    if args.raw_input:
        target_input = Path(args.raw_input)
    elif default_raw.is_file():
        target_input = default_raw
    elif sample_raw.is_file():
        print(f"[NOTICE] Authentic production trace '{default_raw}' not found.")
        print(f"Using development test fixture '{sample_raw}'.")
        target_input = sample_raw
    else:
        print(f"[ERROR] No raw trace or test fixture found in {resources.RAW_DATASET_DIR}")
        sys.exit(1)

    parse_ibm_snia_trace(
        raw_input_path=target_input,
        output_path=Path(args.output) if args.output else None,
        high_priority_ratio=args.priority_ratio,
        trace_len_limit=args.limit,
        seed=args.seed,
    )
