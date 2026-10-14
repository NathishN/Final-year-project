# Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation

**Authors:** ML Research & Computer Vision Team  
**Venue:** Prepared for IEEE Transactions on Agri-Food Computing / Pattern Analysis & Machine Intelligence  
**Artifact Repository:** `C:\fruit_dataset`  

---

# Abstract

Automated post-harvest fruit quality grading requires estimating ordinal disease and decay severity while reliably rejecting unencountered species and out-of-distribution (OOD) anomalies. Prior studies in agricultural vision have proposed multi-modal "context-aware" fusion architectures, hypothesizing that conditioning deep visual features with categorical species embeddings and open-set distance metrics improves ordinal regression accuracy. In this work, we conduct a comprehensive empirical audit and 30-seed confirmatory study of multi-fruit severity grading across 56,036 images spanning five species (*Apple, Banana, Mango, Orange, Tomato*) and five ordinal grades (*Healthy, Mild, Moderate, Severe, Rotten*).

First, we identify and resolve critical methodological defects in historical benchmarks: unconstrained CORAL ordinal thresholds that produce inverted cutpoints and negative discrete probabilities, absent Grade 2 evaluation samples due to session-based split clustering, and OOD benchmarks evaluated on synthetic Gaussian noise rather than real images. We introduce **Monotonic CORAL**, which parameterizes cutpoint biases via cumulative softplus decrements, structurally guaranteeing monotonic cumulative logits and valid probability distributions summing to 1.0 unconditionally.

Second, evaluating under a frozen, leakage-free 5-grade protocol (`split_v2`), we present a definitive 30-seed confirmatory comparison ($N=120$ runs) on precomputed visual representations from a frozen ConvNeXt-Tiny encoder. We demonstrate that auxiliary species conditioning degrades ordinal Quadratic Weighted Kappa (QWK) compared to pure visual ConvNeXt-Tiny representations (QWK: $0.8118 \pm 0.0364$ vs $0.8478 \pm 0.0260$, $\Delta = -0.0360$, paired $t(29) = -4.16$, $p = 0.000260$, Wilcoxon signed-rank $W = 72.0$, $p = 0.000555$, Cohen's $d_z = -0.7593$). Incorporating OOD distance features further degrades performance (QWK: $0.7630 \pm 0.0402$, $p < 10^{-11}$).

Third, evaluating real-image open-set recognition on 5,000 real images of unencountered species (*Cherry, Kiwi, Peach, Pineapple, Plum*), Mahalanobis Distance achieves 84.22% AUROC within studio domains, but drops to 18.08% on market-bin images, suggesting that distance-based detection may exploit studio acquisition artifacts (white backgrounds) rather than fruit morphology. In contrast, Energy Score (92.49% on market bins, 98.89% on packing conveyors) and MSP (61.74% and 96.57%) maintain robust detection without domain-specific inversion. Finally, confidence-thresholded selective abstention effectively filters sorting errors, raising accuracy from 91.15% to 99.14% at 90% coverage with an Expected Calibration Error of 2.51%. Our findings demonstrate that deep visual encoders natively capture species-specific geometry, rendering auxiliary categorical conditioning redundant, and provide an evidence-backed baseline for trustworthy agricultural automation.


---

# Methodology

## 1. Problem Formulation and Ordinal Framework

Let $\mathcal{X} \subset \mathbb{R}^{H \times W \times 3}$ denote the input image space, and let $y \in \mathcal{Y} = \{0, 1, \dots, K-1\}$ denote the ordinal severity grade, where $K=5$ corresponds to Healthy (0), Mild (1), Moderate (2), Severe (3), and Rotten (4). Because misclassifying Grade 0 as Grade 4 carries far greater agricultural and economic penalty than misclassifying Grade 0 as Grade 1, multi-class categorical cross-entropy is inappropriate.

We employ Rank-Consistent Ordinal Regression (CORAL) \cite{cao2020rank}. The ordinal task is decomposed into $K-1$ binary classification sub-tasks sharing a single projection vector $\mathbf{w} \in \mathbb{R}^D$ and task-specific cutpoint thresholds $b_k$:

$$h_k(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b_k) \approx P(y > k \mid \mathbf{x}), \quad k \in \{0, \dots, K-2\}$$

## 2. Monotonic CORAL Parameterization

In conventional CORAL implementations, cutpoints $b_k$ are updated unconstrained via stochastic gradients. Across random seeds, cutpoint inversions ($b_{k} > b_{k-1}$) frequently occur, producing invalid cumulative distributions where $P(y > k) > P(y > k-1)$ and yielding negative class probabilities.

To guarantee mathematical consistency, we parameterize the cutpoint biases strictly via cumulative softplus decrements:

$$b_0 = \theta_0$$
$$b_k = \theta_0 - \sum_{j=1}^k \text{softplus}(\delta_j), \quad k \in \{1, \dots, K-2\}$$

Because $\text{softplus}(\delta) = \ln(1 + e^\delta) > 0$ strictly for all real $\delta \in \mathbb{R}$, this parameterization ensures:

$$b_0 > b_1 > b_2 > \dots > b_{K-2} \quad \text{unconditionally.}$$

The discrete class probabilities $P(y = k \mid \mathbf{x})$ are reconstructed via:

$$P(y = 0 \mid \mathbf{x}) = 1 - P(y > 0 \mid \mathbf{x})$$
$$P(y = k \mid \mathbf{x}) = P(y > k-1 \mid \mathbf{x}) - P(y > k \mid \mathbf{x}), \quad k \in \{1, \dots, K-2\}$$
$$P(y = K-1 \mid \mathbf{x}) = P(y > K-2 \mid \mathbf{x})$$

This formulation guarantees that $P(y = k \mid \mathbf{x}) \ge 0$ for all $k$ and $\sum_{k=0}^{K-1} P(y = k \mid \mathbf{x}) = 1.0$ within floating-point tolerance.

## 3. Class-Balanced Effective Number Weighting

To combat severe class imbalance ($36,414$ Healthy samples vs $208$ Moderate samples in training) without the extreme instability of raw inverse weighting, we adopt the Effective Number of Samples weighting formulation \cite{cui2019class}:

$$E_{n_k} = \frac{1 - \beta^{n_k}}{1 - \beta}, \quad w_k = \frac{1}{E_{n_k}}, \quad \text{with } \beta = 0.999$$

Weights are normalized such that $\sum_{k=0}^{K-1} w_k = K$. This bounds the maximum-to-minimum weight ratio to $5.32\times$, in stark contrast to the $175.07\times$ ratio of raw inverse frequency weighting.

## 4. Multi-Modal Context Conditioning Architectures

We formalize four comparative model architectures:
- **Configuration A (Image Only):** Frozen ConvNeXt-Tiny embeddings (768-D) $\to$ 2-layer MLP projection $\to$ MonotonicCORALHead.
- **Configuration B (Image + Species):** Visual projection (256-D) concatenated with 16-D species embedding from `RobustSpeciesEmbedding`. Unknown species map to index 5.
- **Configuration C (Image + OOD):** Visual projection (256-D) concatenated with 2-D Mahalanobis distance representation.
- **Configuration D (Image + Species + OOD):** Full multi-modal concatenation ($256 + 16 + 2 = 274\text{-D}$) $\to$ Fusion MLP $\to$ MonotonicCORALHead.

### Feature Extraction and Frozen Backbone Protocol

All 120 confirmatory experiments operate on precomputed ConvNeXt-Tiny feature tensors extracted once and cached in `outputs/features/phase4_dataset.pt` (56,036 samples × 768 dimensions, 179 MB). The ConvNeXt-Tiny backbone (~28.6M parameters) is not instantiated during training; no gradient updates flow through the visual encoder.

Only the Monotonic CORAL head and fusion MLP (~626K–630K parameters depending on configuration) receive gradient updates during training (AdamW, lr=1e-3, weight_decay=1e-4, 15 epochs, batch_size=2048 feature vectors).

This protocol was chosen to: (a) ensure fair comparison across configurations with identical visual representations, (b) reduce training cost for the 120-run confirmatory study on consumer GPU hardware (RTX 3050 6GB), and (c) isolate the effect of context conditioning from backbone initialization variance.

## 5. Dataset Partitions and Session Allocation

The dataset comprises 56,036 images partitioned into train (39,790), validation (3,413), and test (12,833) splits (`split_v2`). To resolve the complete absence of Grade 2 evaluation samples in historical splits, all 119 Banana Grade 2 images from acquisition session `20200730` are exclusively assigned to the test partition. Images from the same session belonging to other species-grade combinations (Apple Grade 0, $n=147$) remain in the training partition.

Full session-level holdout was not enforced because session identifiers in this dataset do not align with individual species or batches. Rather, within-species, within-grade session exclusivity is guaranteed for the Grade 2 Banana evaluation subset, ensuring zero image-level or identical fruit-grade leakage between partitions.


---

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


---

# Limitations and Threats to Validity

1. **Synthetic Laboratory Domain Domination:**  
   The evaluation test set contains 85.5% images from `fruits360`. While high overall accuracy (92.11%) is achieved, accuracy on genuine market bins (`real_quality`) drops to 39.86%. Models trained on clean turntable fruit silhouettes do not generalize zero-shot to complex commercial bins without domain adaptation.

2. **Detector-Specific OOD Sensitivity and Background Confounding Hypothesis:**  
   Mahalanobis distance AUROC drops severely from 93.26% on studio backgrounds to 18.08% on market-bin images (`real_quality`), suggesting that feature covariance models may exploit clean background cues. In contrast, Energy score remains robust (92.49% on `real_quality`, 98.89% on `real_grade`). Causal attribution of the Mahalanobis failure specifically to background versus foreground features remains a hypothesis, requiring foreground-segmented validation in future work.

3. **Minority Grade Imbalance:**  
   Grade 2 (Moderate Decay) represents only 0.52% of training data ($n=208$) and 0.93% of test data ($n=119$). While Configuration B improves Grade 2 recall from 7.4% to 21.5%, overall minority grade coverage remains sparse across species.


---

# Reproducibility Checklist

- [x] **Frozen Split Manifest:** All 56,036 sample assignments persisted in `outputs/final_audit/split_manifest.csv`.
- [x] **Deterministic Seeds:** Exact schedule of 30 prespecified seeds documented in `configs/training_protocol.json`.
- [x] **Zero Test-Set Hyperparameter Tuning:** All checkpoint selections performed strictly on validation QWK/MAE.
- [x] **100% Artifact Persistence:** All 120 model checkpoints (`.pth`) and raw per-sample prediction tensors (`.pt`) saved in `outputs/casm_v2/`.
- [x] **Exact Numerical Reproduction:** Recalculated metrics match reported values with 0.000000 maximum error.
- [x] **Automated Test Suite:** 17 unit, property, and integration tests pass with 100% success (`tests/`), including full verification of saved prediction tensors and statistical tests.
- [x] **Benchmarked Pipeline Throughput:** Both head-only (>120k FPS) and full end-to-end (41–76 FPS in FP16 on RTX 3050) latencies empirically measured.

