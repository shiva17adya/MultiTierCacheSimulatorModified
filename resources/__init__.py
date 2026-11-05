import os
from pathlib import Path
from typing import List

# Base resources directory
RESOURCES_DIR = Path(__file__).resolve().parent
PATH = str(RESOURCES_DIR)

# Standard dataset directory paths
SYNTHETIC_DIR = RESOURCES_DIR / "dataset_synthetic"
JEDI_DIR = RESOURCES_DIR / "dataset_jedi"
SNIA_DIR = RESOURCES_DIR / "dataset_snia"
RAW_DATASET_DIR = RESOURCES_DIR / "raw_dataset" / "raw"
DATASETS_DIR = RESOURCES_DIR / "datasets"
EASY_DIR = RESOURCES_DIR / "easy"

# Automatically ensure required directory hierarchy exists
for d in [SYNTHETIC_DIR, JEDI_DIR, SNIA_DIR, RAW_DATASET_DIR, DATASETS_DIR, EASY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IGNORED_EXTENSIONS = {".tgz", ".sh", ".bat", ".gz", ".py", ".pyc"}


def find_trace_files(directory: Path) -> List[str]:
    """Find valid trace CSV/data files in the specified directory."""
    if not directory.exists():
        return []
    return [
        str(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() not in IGNORED_EXTENSIONS and not p.name.startswith(".")
    ]


def get_dataset_files(category: str = "synthetic") -> List[str]:
    """Retrieve trace files for a specific category: 'synthetic', 'jedi', 'snia', or 'datasets'."""
    category_map = {
        "synthetic": SYNTHETIC_DIR,
        "jedi": JEDI_DIR,
        "snia": SNIA_DIR,
        "raw": RAW_DATASET_DIR,
        "datasets": DATASETS_DIR,
        "easy": EASY_DIR,
    }
    target_dir = category_map.get(category.lower(), SYNTHETIC_DIR)
    return find_trace_files(target_dir)


# Default NDN_PACKETS: prioritize synthetic if present, otherwise fallback to datasets directory
NDN_PACKETS = get_dataset_files("synthetic")
if not NDN_PACKETS:
    NDN_PACKETS = get_dataset_files("datasets")
