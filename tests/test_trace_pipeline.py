import csv
from pathlib import Path
import pytest
import sys

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import resources
from traces.trace_creating_and_parsing.synthetic_trace import generate_synthetic_trace
from traces.trace_creating_and_parsing.jedi_trace import generate_synthetic_jedi_workload
from traces.trace_creating_and_parsing.snia_trace import parse_ibm_snia_trace
from traces.trace_reading.arc_trace import ARCTrace
from traces.trace_reading.common_trace import CommonTrace
from traces.trace_reading.priority_trace import PriorityTrace


def test_resources_import_and_directory_creation():
    """Verify that importing resources automatically creates standard directories and does not raise FileNotFoundError."""
    assert resources.RESOURCES_DIR.is_dir()
    assert resources.SYNTHETIC_DIR.is_dir()
    assert resources.JEDI_DIR.is_dir()
    assert resources.SNIA_DIR.is_dir()
    assert resources.RAW_DATASET_DIR.is_dir()
    assert resources.DATASETS_DIR.is_dir()
    assert resources.EASY_DIR.is_dir()
    assert isinstance(resources.NDN_PACKETS, list)


def test_synthetic_trace_generation(tmp_path):
    """Validate synthetic Zipf-like trace generation properties."""
    req_count = 5000
    out_file = tmp_path / "test_synth.csv"

    generated = generate_synthetic_trace(
        n_unique_items=1000,
        zipf_alpha=1.0,
        high_priority_ratio=0.20,
        num_requests=req_count,
        poisson_lambda=10.0,
        min_data_size=200,
        max_data_size=8192,
        seed=100,
        output_path=out_file,
    )
    assert generated.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == req_count

    h_count = sum(1 for r in rows if r[4] == "h")
    l_count = sum(1 for r in rows if r[4] == "l")
    assert h_count + l_count == req_count
    # Priority should be approximately 20% (allow tolerance for binomial distribution)
    hp_ratio = h_count / req_count
    assert 0.17 <= hp_ratio <= 0.23

    # Check timestamps and sizes
    prev_ts = -1.0
    for r in rows:
        assert r[0] == "d"
        ts = float(r[1])
        assert ts >= prev_ts
        prev_ts = ts
        size = int(r[3])
        assert 200 <= size <= 8192
        assert r[4] in ("h", "l")
        assert float(r[5]) == 4.0  # InterestLifetime
        rtt = float(r[6])
        assert 0.01 <= rtt <= 0.20


def test_reproducibility_with_seed(tmp_path):
    """Verify that using identical seeds generates exact byte-for-byte identical output."""
    file_a = tmp_path / "run_a.csv"
    file_b = tmp_path / "run_b.csv"
    file_diff = tmp_path / "run_diff.csv"

    generate_synthetic_trace(num_requests=2000, seed=999, output_path=file_a)
    generate_synthetic_trace(num_requests=2000, seed=999, output_path=file_b)
    generate_synthetic_trace(num_requests=2000, seed=111, output_path=file_diff)

    assert file_a.read_text(encoding="utf-8") == file_b.read_text(encoding="utf-8")
    assert file_a.read_text(encoding="utf-8") != file_diff.read_text(encoding="utf-8")


def test_jedi_synthetic_approximation(tmp_path):
    """Validate synthetic Jedi video trace generation."""
    req_count = 3000
    out_file = tmp_path / "test_jedi.csv"

    generate_synthetic_jedi_workload(
        n_unique_items=5000,
        num_requests=req_count,
        arrival_rate=10.0,
        high_priority_ratio=0.20,
        seed=42,
        output_path=out_file,
    )
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == req_count
    h_count = sum(1 for r in rows if r[4] == "h")
    hp_ratio = h_count / req_count
    assert 0.17 <= hp_ratio <= 0.23
    for r in rows:
        assert int(r[3]) <= 8192
        assert r[4] in ("h", "l")


def test_ibm_snia_parser_with_fixture(tmp_path):
    """Validate SNIA IBM trace parser on the test fixture."""
    fixture_path = resources.RAW_DATASET_DIR / "sample_IBMObjectStoreTrace001Part0.txt"
    assert fixture_path.is_file()

    out_file = tmp_path / "parsed_ibm.csv"
    parse_ibm_snia_trace(
        raw_input_path=fixture_path,
        output_path=out_file,
        high_priority_ratio=0.20,
        trace_len_limit=100,
        seed=42,
    )
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # The sample fixture contains exactly 22 REST.GET.OBJECT requests
    assert len(rows) == 22
    for r in rows:
        assert r[0] == "d"
        assert r[2].startswith("ibm_obj_")
        assert int(r[3]) > 0
        assert r[4] in ("h", "l")


def test_trace_readers_compatibility():
    """Verify that existing Trace reader classes load the generated synthetic traces without errors."""
    synth_files = resources.get_dataset_files("synthetic")
    assert len(synth_files) > 0, "At least one synthetic trace should be present in resources/dataset_synthetic"

    # Test ARCTrace
    arc_trace = ARCTrace()
    arc_trace.gen_data(trace_len_limit=50)
    assert len(arc_trace.data) == 50
    assert len(arc_trace.data[0]) == 7

    # Test CommonTrace
    common_trace = CommonTrace()
    common_trace.gen_data(trace_len_limit=50)
    assert len(common_trace.data) == 50

    # Test PriorityTrace
    priority_trace = PriorityTrace()
    priority_trace.gen_data(trace_len_limit=50)
    assert len(priority_trace.data) == 50
