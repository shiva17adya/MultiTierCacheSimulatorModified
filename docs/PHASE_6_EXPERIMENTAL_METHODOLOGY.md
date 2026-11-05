# Phase 6 Experimental Methodology: Comprehensive Evaluation of HA-QM-ARC

**Project:** Enhanced QM-ARC: A Workload-Aware, QoS-Based Multi-Tier Adaptive Cache Replacement Simulator  
**Proposed Algorithm:** History-Aware Adaptive QM-ARC (HA-QM-ARC)  
**Date:** 2026-11-05  

---

## 1. Overview and Evaluation Objectives

This document establishes the experimental methodology for the comprehensive comparative evaluation of **HA-QM-ARC (History-Aware Adaptive QM-ARC)** against all seven original cache replacement policies implemented in the `MultiTierCacheSimulator` framework.

The primary objective is an empirical, statistically sound evaluation to determine:
1. Whether HA-QM-ARC preserves competitive performance under stationary, stable workloads without regressions.
2. Whether HA-QM-ARC mitigates sequential scan pollution and preserves resident frequency items in $T_2$.
3. Whether history-aware momentum ($\Delta p_{\text{history}}$) accelerates adaptation speed during non-stationary workload phase shifts.
4. Whether dynamic QoS placement ($\lambda_{\text{effective}}$) reduces high-priority SLA penalties during high-priority traffic surges, and at what cost to low-priority traffic.
5. The isolated performance contributions of each architectural component via ablation.
6. The sensitivity of the algorithm to history window size ($W$) and controller hyperparameters ($\omega_{\text{burst}}, \omega_{\text{scan}}, \omega_{\text{grad}}, \kappa_{\text{qos}}$).

---

## 2. Algorithms Under Evaluation

All eight registered cache replacement algorithms are evaluated:

| Algorithm Name | Category | Coordinator Class | Tier Policy Class | Description |
| :--- | :---: | :---: | :---: | :--- |
| **LRU** | standard | None | `LRUPolicy` | Classic Least Recently Used eviction |
| **LFU** | standard | None | `LFUPolicy` | Classic Least Frequently Used eviction |
| **Random** | standard | None | `RandPolicy` | Uniform random eviction baseline |
| **Priority-LRU** | priority_routing | None | `LRUPolicy` | Priority-partitioned routing (HP to DRAM, LP to SSD) |
| **M-ARC** | arc | `AbstractMARCPolicy` | `MARCPolicy` | Multi-tier Adaptive Replacement Cache without QoS |
| **QM-ARC** | arc | `AbstractQMARCPolicy` | `QMARCPolicy` | Original QoS-aware Multi-tier ARC (Ait-Oucheggou et al.) |
| **QL-QM-ARC** | arc | `AbstractQLQMARCPolicy` | `QLQMARCPolicy` | Q-Learning enhanced QM-ARC |
| **HA-QM-ARC** | arc | `AbstractHAQMARCPolicy` | `TierHAQMARCPolicy` | Proposed History-Aware Adaptive QM-ARC |

---

## 3. Workload Suite and Generation Protocols

All comparisons are **paired**: for any given random seed and workload configuration, every evaluated algorithm processes the **exact same request sequence** with identical timestamps, item sizes, priorities, and arrival intervals.

### A. Workload 1: Stable Zipf ($\alpha = 1.0$)
- Catalog: 10,000 unique objects.
- Skewness: Zipf distribution with $\alpha = 1.0$.
- Priority Split: 20% high priority (HPC), 80% low priority (LPC).
- Length: 1,000 requests per run.
- Seeds: 5 independent seeds ($42, 123, 456, 789, 999$).

### B. Workload 2: Sequential Scan Pollution
- Phase 1 (Pre-Scan, 400 requests): High-frequency access to a working set of 50 hot items ($P_{\text{hot}} = 0.90$).
- Phase 2 (Scan Burst, 400 requests): Strictly sequential scan of previously unseen unique items (`video_chunk_1000` to `video_chunk_1399`).
- Phase 3 (Post-Scan Recovery, 400 requests): Return to the Phase 1 hot working set.
- Length: 1,200 continuous requests.

### C. Workload 3: Workload Phase Change (Non-Stationary Popularity Shift)
- Phase A (Requests 1–500): Skewed Zipf distribution over working set $A$ (items $0\dots 999$).
- Phase B (Requests 501–1,000): Shift to working set $B$ (items $5,000\dots 5,999$).
- Phase C (Requests 1,001–1,500): Return to working set $A$ (items $0\dots 999$).
- Length: 1,500 continuous requests.

### D. Workload 4: High-Priority Traffic Surge (QoS Stress)
- Phase 1 (Requests 1–500): 20% HP, 80% LP (Normal traffic).
- Phase 2 (Requests 501–1,000): 70% HP, 30% LP (Surge traffic).
- Phase 3 (Requests 1,001–1,500): 20% HP, 80% LP (Recovery traffic).
- Length: 1,500 continuous requests.

---

## 4. Hardware and Storage Tier Configurations

### A. Two-Tier Storage Hierarchy (DRAM + SSD)
- Ratio: DRAM:SSD = 1:5.
- DRAM: Latency $10^{-7}$ s (100 ns), Read throughput 40 GB/s, Write throughput 20 GB/s.
- SSD: Latency $10^{-5}$ s (10 $\mu$s), Read throughput 2 GB/s, Write throughput 1 GB/s.
- Evaluated Cache Proportions: 0.5%, 1.0%, 5.0%, 10.0% of catalog.

### B. Three-Tier Storage Hierarchy (DRAM + SSD + NVMe)
- Ratio: DRAM:SSD:NVMe = 1:5:25 (sum = 31 units).
- NVMe: Latency $10^{-4}$ s (100 $\mu$s), Read throughput 500 MB/s, Write throughput 250 MB/s.
- Evaluated Cache Proportions: 1.0%, 5.0%, 10.0% of catalog.

---

## 5. Performance Metrics

To ensure mathematical consistency and avoid ambiguous terminology:

1. **Global Hit Ratio (GHR)**:
   $$\text{GHR} = \frac{\sum_{i=1}^{M} \text{Hits}_i}{N}$$
   where $\text{Hits}_i$ is the number of hits in physical tier $i$, and $N$ is total requests.
2. **High-Priority Hit Ratio ($\text{HR}_{\text{HP}}$)**:
   $$\text{HR}_{\text{HP}} = \frac{\sum_{i=1}^{M} \text{Hits}_{\text{HP},i}}{N_{\text{HP}}}$$
3. **Low-Priority Hit Ratio ($\text{HR}_{\text{LP}}$)**:
   $$\text{HR}_{\text{LP}} = \frac{\sum_{i=1}^{M} \text{Hits}_{\text{LP},i}}{N_{\text{LP}}}$$
4. **SLA Penalties ($\text{Penalty}_{\text{HP}}, \text{Penalty}_{\text{LP}}, \text{Penalty}_{\text{Total}}$)**:
   Direct accumulated SLA penalties from miss latency and tier retrieval times using the paper's QoS penalty functions.
5. **Conditional Tier Hit Ratio vs Hit Contribution**:
   - **Contribution**: $\text{Hits}_i / N$ (fraction of total requests satisfied by tier $i$).
   - **Conditional**: $\text{Hits}_i / \text{Requests Reaching Tier } i$.
6. **Average Retrieval Latency**: Average time required to retrieve requested objects across tiers.
7. **Computational Overhead**: Wall-clock simulation time and per-request execution microseconds.

---

## 6. Statistical Methodology

1. **Paired Observations**: Every metric comparison between HA-QM-ARC and a baseline algorithm $B$ is paired per random seed $s$:
   $$D_s = X_{\text{HA-QM-ARC}, s} - X_{B, s}$$
2. **Descriptive Statistics**: Report Mean, Standard Deviation (SD), and 95% Confidence Interval (CI) using Student's $t$-distribution ($n=5, \text{df}=4, t_{0.975} = 2.776$):
   $$\text{CI}_{95\%} = \bar{X} \pm t_{0.975} \cdot \frac{\text{SD}}{\sqrt{n}}$$
3. **Hypothesis Testing**: Two-tailed paired Student's $t$-test.
4. **Effect Size (Cohen's $d_z$)**:
   $$d_z = \frac{\bar{D}}{\text{SD}_D}$$
   Thresholds: Small ($|d| \ge 0.2$), Medium ($|d| \ge 0.5$), Large ($|d| \ge 0.8$).
5. **Scientific Rigor**: Acknowledge sample size constraints ($n=5$). Report all outcomes objectively without suppressing neutral or negative findings.
