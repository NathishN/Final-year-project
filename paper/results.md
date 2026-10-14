# Experimental Results

## 1. Primary Confirmatory Hypothesis: Species Conditioning Degradation

We executed 30 prespecified seeds across all four configurations ($N=120$ total independent runs) using identical training budgets (15 epochs, AdamW optimizer, lr=$10^{-3}$, batch size 2,048) and validation-only checkpoint selection.

| Configuration | Architecture | Test QWK (Mean ± Std) | Test MAE | Test Accuracy (%) | OBO Acc (%) | Macro-F1 (%) | Grade 2 Recall (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A** | Image Only | **0.8478 ± 0.0260** | **0.1145 ± 0.0178** | **92.11%** | **97.30%** | **49.48%** | 7.4% |
| **Config B** | Image + Species | 0.8118 ± 0.0364 | 0.1268 ± 0.0184 | 92.06% | 96.56% | 49.35% | **21.5%** |
| **Config C** | Image + OOD | 0.7778 ± 0.0450 | 0.1554 ± 0.0241 | 89.41% | 95.61% | 42.39% | 6.6% |
| **Config D** | Image + Species + OOD | 0.7630 ± 0.0402 | 0.1536 ± 0.0170 | 90.29% | 95.34% | 44.39% | 12.7% |

### Statistical Contrast Analysis (Paired over 30 Prespecified Seeds)

$$\Delta\text{QWK}_{B - A} = -0.0360, \quad 95\%\text{ CI: } [-0.0537, -0.0183]$$
- Paired Two-Sided $t$-Test: $t(29) = -4.1590, \quad \mathbf{p = 0.000260}$
- Wilcoxon Signed-Rank Test: $W = 72.0$ (`zero_method='wilcox'`), $\mathbf{p = 0.000555}$
- Paired Effect Size: Cohen's $d_z = -0.7593$ (Substantial negative effect)

$$\Delta\text{QWK}_{C - A} = -0.0700, \quad 95\%\text{ CI: } [-0.0902, -0.0498], \quad p = 8.41 \times 10^{-8}, \quad d_z = -1.2947$$
$$\Delta\text{QWK}_{D - A} = -0.0848, \quad 95\%\text{ CI: } [-0.1008, -0.0687], \quad p = 1.13 \times 10^{-11}, \quad d_z = -1.9713$$

## 2. Real-Image Open-Set Recognition Benchmark

Evaluating against 5,000 real images of unseen fruit species (*Cherry, Kiwi, Peach, Pineapple, Plum*) extracted via the official ConvNeXt backbone:

| Detector | AUROC (%) | 95% Bootstrap CI | AUPR (OOD=1) (%) | FPR @ 95% TPR (%) |
|:---|---:|:---:|---:|---:|
| **Mahalanobis Distance** | **84.22%** | [83.7%, 84.8%] | **53.28%** | **38.39%** |
| **Energy Score** | **81.99%** | [81.4%, 82.6%] | 49.96% | 41.38% |
| **MSP (Max Softmax)** | **74.26%** | [73.6%, 75.0%] | 43.16% | 47.85% |

### Per-Species Discrimination:
- **Pineapple:** 93.00% AUROC
- **Plum:** 88.08% AUROC
- **Kiwi:** 84.77% AUROC
- **Peach:** 78.28% AUROC
- **Cherry:** 76.99% AUROC

### Domain-Stratified OOD Detection Breakdown:

| In-Distribution Domain | Sample Count ($n$) | Mahalanobis AUROC | Energy Score AUROC | MSP AUROC |
|:---|---:|---:|---:|---:|
| `fruits360` (Laboratory Studio) | 10,974 | 93.26% | 79.99% | 75.21% |
| `real_severity` (Field Photography) | 312 | 94.10% | 98.67% | 95.30% |
| `real_quality` (Market Bins & Crates) | 1,480 | **18.08%** | 92.49% | 61.74% |
| `real_grade` (Packing Conveyor) | 67 | **18.81%** | **98.89%** | **96.57%** |

> **Detector-Specific Observation:** While Mahalanobis distance detection collapses to 18.08% on natural grocery acquisitions, Energy Score achieves 92.49% on `real_quality` and 98.89% on `real_grade`. This demonstrates that open-set failure under natural backgrounds is detector-specific rather than universal, indicating feature covariance distortion under non-studio backgrounds.

## 3. Domain Transfer Gap Analysis

| Domain | Partition Type | Test Samples | Accuracy (%) | MAE | QWK |
|:---|:---|---:|---:|---:|:---:|
| `fruits360` | Laboratory Studio Turntable | 10,974 | **99.88%** | 0.0012 | Undefined (Variance=0) |
| `real_severity` | Field Photography | 312 | **82.69%** | 0.1731 | 0.8980 |
| `real_quality` | Market Bins & Crates | 1,480 | **39.86%** | 0.8135 | 0.5920 |
| `real_grade` | Packing Station Conveyor | 67 | **19.40%** | 1.1642 | 0.2800 |
| **`split_real` (Combined)** | **All Real-World Data** | **1,859** | **46.32%** | **0.7187** | **0.7045** |

## 4. Calibration & Selective Prediction

- **Expected Calibration Error (10 Bins):** 2.51% (mean across confirmatory seeds; representative seed 83 achieves 1.27%). Pooled ECE is primarily weighted by studio images (85.5% of test samples).
- **Brier Score:** 0.0887
- **Selective Abstention:** Discarding the lowest 10% of confident predictions raises sorting accuracy from 91.15% to **99.14%**, and reduces MAE from 0.1382 to **0.0166**.

## 5. Computational Efficiency and Parameter Census

- **Trainable Parameters:** 626,180 (Config A) to 630,884 (Config D). The ConvNeXt-Tiny visual backbone (27,823,973 parameters) remains frozen.
- **Head-Only Throughput:** >100,000 samples/sec on CPU and >300,000 samples/sec on RTX 3050 GPU at batch size 2,048 using cached 768-D feature vectors.
- **Full End-to-End Pipeline Throughput (Raw Image $\to$ Backbone $\to$ Head):** On consumer GPU hardware (NVIDIA RTX 3050 Laptop GPU, FP16), the complete pipeline achieves **41.0 FPS** (latency 24.4 ms) at batch size 1 and **74.6 to 76.2 FPS** (latency 13.1–13.4 ms/image) at batch sizes 16–32, exceeding standard commercial sorting conveyor speeds (10–30 items/sec).
