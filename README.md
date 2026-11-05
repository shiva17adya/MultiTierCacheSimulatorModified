# Multi-Tier Cache Simulator for NDN (QM-ARC & QL-QM-ARC)

> A reproducible Python simulator for evaluating QoS-aware cache-replacement policies across two- and three-tier NDN caches.

This repository contains the simulation framework for the research paper:
> **"QM-ARC: QoS-aware Multi-tier Adaptive Cache Replacement Strategy"**  
> *Lydia Ait-Oucheggou, Stéphane Rubini, Abdella Battou, Jalil Boukhobza* (Elsevier FGCS / NIST / Lab-STICC)

The simulator models heterogeneous multi-tier caching nodes (e.g., DRAM, Optane/NVMe SSD, and Disk) evaluating algorithms including:
- **QM-ARC:** QoS-aware Multi-tier Adaptive Replacement Cache.
- **QL-QM-ARC:** Q-learning enabled multi-tier adaptive cache.
- **M-ARC:** Multi-tier ARC baseline without QoS priority management.
- **Priority-LRU:** Multi-queue priority-partitioned LRU.
- **Classic Baselines:** LRU, LFU, and Random.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14.3 on Windows 11).
- PowerShell or command shell with Git installed.

### Virtual Environment Initialization
From the repository root directory (`MultiTierCacheSimulator`):

```powershell
# Create a dedicated virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate
```

### Dependency Installation
Install verified, compatible dependencies:

```powershell
pip install -r requirements.txt
```

The updated `requirements.txt` specifies:
```text
simpy>=4.1.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pytest>=8.0.0
```
*(Note: Legacy `cryptography` and `pyndn` packages listed in earlier iterations are completely unused by the simulator and have been removed).*

---

## 2. Dataset Directory Structure

The project automatically initializes the required `resources` directory tree:

```
resources/
├── dataset_synthetic/          # Generated Zipf-like synthetic trace CSVs
├── dataset_jedi/               # Jedi CDN video workload CSVs (synthetic or raw)
├── dataset_snia/               # Parsed SNIA IBM Object Store trace CSVs
├── raw_dataset/
│   └── raw/                    # Raw external trace files and fixtures
│       ├── sample_IBMObjectStoreTrace001Part0.txt  # Development test fixture
│       └── jedi_w             # (Optional) Authentic raw Jedi trace
├── datasets/                   # Sample verification datasets (lru_better.csv)
├── easy/                       # Reference fixtures (lfu_better.csv)
└── __init__.py                 # Safe dataset loader and cross-platform path resolver
```

`resources/__init__.py` uses `pathlib.Path` to guarantee cross-platform compatibility and avoids import-time `FileNotFoundError` exceptions by auto-creating directories on demand.

---

## 3. Workload Generation & Trace Preparation

All trace generators produce standard NDN packet CSV rows:
```
data_back,timestamp,name,size,priority,InterestLifetime,response_time
```
- `data_back`: Always `"d"`.
- `timestamp`: Arrival time in seconds (Poisson process).
- `name`: Unique object identifier.
- `size`: Object size in bytes.
- `priority`: `"h"` for High Priority Data (20%), `"l"` for Low Priority Data (80%).
- `InterestLifetime`: NDN interest lifetime in seconds (default `4.0`).
- `response_time`: Simulated retrieval/RTT delay in seconds ($0.01\text{s} - 0.20\text{s}$).

### 3.1 Synthetic Zipf-like Workload Generator
Generates a workload drawn from a Zipf distribution matching the research paper:
- 10,000 unique catalog objects.
- Zipf parameter $\alpha = 1.0$.
- 20% high-priority requests, 80% low-priority requests.
- Poisson arrival rate of 10 requests/second.
- Object sizes between 100 bytes and 8 KB (NDN maximum packet payload).
- Fully reproducible using a fixed random seed.

**Execution Command:**
```powershell
python traces/trace_creating_and_parsing/synthetic_trace.py --items 10000 --alpha 1.0 --priority-ratio 0.20 --requests 100000 --rate 10.0 --seed 42
```
Output is stored at: `resources/dataset_synthetic/synthetic-items10000_alpha1.0_hp0.2_reqs100000_lambda10.csv`.

---

### 3.2 Jedi Video Workload
Models an Akamai production CDN video workload:
- 40,000 unique video chunk objects.
- Heavy-tailed popularity distribution.
- 10 requests/second arrival rate.
- 20% high priority content.

**Execution Command:**
```powershell
python traces/trace_creating_and_parsing/jedi_trace.py --items 40000 --requests 100000 --rate 10.0 --seed 42
```
- If an authentic `jedi_w` raw file is present in `resources/raw_dataset/raw/`, it will parse the real trace.
- Otherwise, it generates an explicitly labeled synthetic approximation:  
  `resources/dataset_jedi/jedi_video_synthetic_approx_items40000_hp0.2_reqs100000.csv`.

---

### 3.3 SNIA IBM Object Store Trace
Parses real-world object storage workloads released by IBM Research on the SNIA IOTTA repository (Trace 36308 / 001 Part 0).

#### Downloading the Authentic Dataset
1. Visit the SNIA IOTTA repository:  
   [http://iotta.snia.org/traces/key-value/36305?only=36308](http://iotta.snia.org/traces/key-value/36305?only=36308)
2. Download `IBMObjectStoreTrace001Part0` (or `IBMObjectStoreTrace000Part0`).
3. Place the uncompressed trace file into:
   ```
   resources/raw_dataset/raw/IBMObjectStoreTrace001Part0
   ```

#### Parsing the Trace
Execute the parser to extract `REST.GET.OBJECT` entries with simulated QoS priority tagging:
```powershell
python traces/trace_creating_and_parsing/snia_trace.py --limit 100000 --priority-ratio 0.20 --seed 42
```
- If the production trace is present, it parses it into `resources/dataset_snia/`.
- If absent, it automatically uses the included test fixture:  
  `resources/raw_dataset/raw/sample_IBMObjectStoreTrace001Part0.txt`.

---

## 4. Run an Experiment

Run a small, deterministic experiment after installing the dependencies:

```powershell
python run_experiment.py --algorithm QM-ARC --workload synthetic --tiers 2 --requests 1000 --seed 42 --plot
```

The command writes machine-readable metrics to `results/` (JSON and CSV) and, when `--plot` is supplied, creates comparison figures. Use `python run_experiment.py --help` for every option and supported policy name.

### Reproduce the Paper Experiments

The paper-reproduction entry point runs the capacity, penalty-lambda, and three-tier sweeps. Start with a small run, then increase the request count for a full study:

```powershell
python reproduce_paper.py --requests 1000 --workloads synthetic --experiments 1 2 3 --seeds 42
```

Methodology, parameter choices, and interpretation guidance are in [docs/PHASE_6_EXPERIMENTAL_METHODOLOGY.md](docs/PHASE_6_EXPERIMENTAL_METHODOLOGY.md).

## 5. Running Validation Tests

A comprehensive unit test suite validates the trace pipeline, directory management, priority distributions, and reproducibility:

```powershell
pytest -q
```

### Verified Test Cases
1. `test_resources_import_and_directory_creation`: Ensures directory auto-creation and safe resource imports.
2. `test_synthetic_trace_generation`: Verifies row count, 20/80 priority ratio, size boundaries ($\le 8\text{ KB}$), and timestamps.
3. `test_reproducibility_with_seed`: Validates byte-identical trace generation with identical seeds.
4. `test_jedi_synthetic_approximation`: Confirms Jedi-style workload generation and column integrity.
5. `test_ibm_snia_parser_with_fixture`: Confirms `REST.GET.OBJECT` filtering on the SNIA fixture.
6. `test_trace_readers_compatibility`: Ensures `ARCTrace`, `CommonTrace`, and `PriorityTrace` load traces cleanly.

---

## 6. Running the Legacy Simulation Suite

To run the simulation suite across all 7 policies:
```powershell
python main.py
```
Output results and log traces will be written to `logs/<timestamp>/` and plotted via `plots/plot_creation.py`.

## Project Layout

```
├── policies/                 # QM-ARC, QL-QM-ARC, HA-QM-ARC, M-ARC and baselines
├── experiments/              # Configurable runners, metrics, plots, and reproduction suite
├── traces/                   # Synthetic, Jedi, and SNIA trace generators/readers
├── resources/                # Lightweight fixtures and generated workload locations
├── tests/                    # Unit and integration tests
├── docs/                     # Experimental methodology
├── run_experiment.py         # Single-experiment CLI
└── reproduce_paper.py        # Paper-reproduction CLI
```

## Results and Data Policy

Generated experiment outputs are reproducible and may be regenerated with the commands above. The repository includes compact result summaries, configurations, and figures; verbose simulator logs remain untracked to keep clones fast. Do not commit licensed third-party trace data—download it separately and place it under `resources/raw_dataset/raw/`.

## Contributing

Please keep experiments deterministic by supplying a seed, add or update tests for behavior changes, and run `pytest -q` before opening a pull request.

## Citation and Attribution

This implementation is based on *QM-ARC: QoS-aware Multi-tier Adaptive Cache Replacement Strategy* by Lydia Ait-Oucheggou, Stéphane Rubini, Abdella Battou, Jalil Boukhobza. Please cite the original work when using this simulator in academic work.
