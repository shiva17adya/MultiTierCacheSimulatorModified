# Final Validated Baseline — QM-ARC Paper Reproduction Suite

## Overview
This directory contains the final, fully validated experimental baseline reproducing the empirical evaluation of the research paper:
> **"QM-ARC: QoS-aware Multi-tier Adaptive Cache Replacement Strategy"**  
> *Lydia Ait-Oucheggou, Stéphane Rubini, Abdella Battou, Jalil Boukhobza.*  
> Elsevier Future Generation Computer Systems (FGCS).

All seven caching algorithms are evaluated under strict configuration isolation, verified physical tier capacity constraints, and corrected research-fidelity metrics.

---

## Directory Structure
- `configs/`: JSON configuration files for every executed experiment.
- `raw/json/`: Raw JSON simulation telemetry records containing full per-tier event counters, hit metrics, latency, and SLA penalties.
- `aggregated/`:
  - `all_experiments_raw.csv`: Consolidated table of all individual simulation runs.
  - `experiments_aggregated_mean.csv`: Aggregated means grouped by `(algorithm, workload, requests, num_tiers, cache_proportion, penalty_lambda)`.
- `plots/`: High-resolution (300 DPI) publication-ready plots for Experiments 1, 2, and 3.
- `reports/`:
  - `table_3_penalty_comparison.csv`: Replicated Table 3 showing penalty percentage differences relative to each strategy using the paper's exact formula:
    $$\text{Diff} = \frac{P_{\text{QM-ARC}} - P_{\text{Strategy}}}{P_{\text{Strategy}}} \times 100\%$$
- `statistical/`:
  - `statistical_trials_raw.csv`: Raw data across 5 independent random seeds ($N=1000$).
  - `statistical_analysis_summary.csv`: Statistical report containing Mean, Standard Deviation, 95% Confidence Intervals (Student's $t$, $df=4$), and Cohen's $d$ effect sizes.

---

## Algorithms Evaluated
1. **QM-ARC**: QoS-aware Multi-tier Adaptive Replacement Cache (Proposed paper policy).
2. **QL-QM-ARC**: Q-Learning extension with state space $S = (p, \text{cache\_event})$ and $\epsilon$-greedy exploration decay.
3. **M-ARC**: Multi-tier ARC without QoS priority positioning (ablation baseline).
4. **Priority-LRU**: Multi-queue priority LRU (assigned per-tier; excluded from 3-tier per paper Sec 4.2.3).
5. **LRU**: Least Recently Used across tiers.
6. **LFU**: Least Frequently Used across tiers.
7. **Random**: Random eviction across tiers.

---

## Workload Authenticity & Limitations
- **Synthetic Zipf (`synthetic_zipf_alpha1.0_hp0.2`)**: **Authentic Synthetic Workload** conforming strictly to the mathematical specification in Section 4.1.3 ($\alpha=1.0$, 10,000 unique catalog objects, 20% high priority, 80% low priority, size up to 8KB).
- **IBM Object Store (`ibm_object_store_fixture`)**: **Development / Test Fixture** derived from the SNIA IOTTA IBM Object Store Trace 000 Part 0 format. Production multi-GB raw logs were not distributed in the repository.
- **Jedi CDN (`jedi_video_synthetic_approx`)**: **Synthetic Approximation** modeled after Akamai video CDN traces (40,000 unique items) using in-tree parameters.

---

## Mathematical Formulations
- **Tier Capacities**:
  - Two-tier ($1:5$ target): $c_1 = \max(1, \text{round}(T / 6)),\ c_2 = T - c_1$
  - Three-tier ($1:5:25$ target): $c_1 = \max(1, \text{round}(T / 31)),\ c_2 = \max(5 c_1, \text{round}(5T / 31)),\ c_3 = T - c_1 - c_2$
- **Table 3 Formula**:
  $$\text{Diff}(\%) = \frac{P_{\text{QM-ARC}} - P_{\text{Strategy}}}{P_{\text{Strategy}}} \times 100\%$$
  *(Negative values indicate QM-ARC achieved a lower penalty / superior QoS performance).*
