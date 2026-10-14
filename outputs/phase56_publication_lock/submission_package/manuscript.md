# Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation

**Authors:** ML Research & Computer Vision Team  
**Venue:** Prepared for IEEE Transactions on Agri-Food Computing / Pattern Analysis & Machine Intelligence  
**Artifact Repository:** `[Repository Archive Provided with Submission]` (Open Source Data & Code Archive)  
**Publication Status:** Phase 5.6 Locked & Certified Submission Package  

---

# Abstract

Automated post-harvest fruit quality grading requires estimating ordinal disease and decay severity while reliably rejecting unencountered species and out-of-distribution (OOD) anomalies. Prior studies in agricultural vision have proposed multi-modal "context-aware" fusion architectures, hypothesizing that conditioning deep visual features with categorical species embeddings and open-set distance metrics improves ordinal regression accuracy. In this work, we conduct a comprehensive forensic audit and 30-seed confirmatory study of multi-fruit severity grading across 56,036 images spanning five species (*Apple, Banana, Mango, Orange, Tomato*) and five ordinal grades (*Healthy, Mild, Moderate, Severe, Rotten*).

First, we resolve critical methodological defects in historical benchmarks: unconstrained CORAL ordinal thresholds that produce inverted cutpoints and negative discrete probabilities, absent Grade 2 evaluation samples due to session-based split clustering, and OOD benchmarks evaluated on synthetic Gaussian noise rather than real images. We introduce **Monotonic CORAL**, which parameterizes cutpoint biases via cumulative softplus decrements, structurally guaranteeing monotonic cumulative logits and valid probability distributions summing to 1.0 unconditionally.

Second, evaluating under both a historical protocol (`split_v2`, $N_{\text{test}}=12,833$, paired seeds $n=30$) and a strict zero-leakage 100% group session-held-out protocol ($N_{\text{test}}=12,980$, paired seeds $n=30$), we present a definitive confirmatory comparison ($N=240$ total model runs) on visual representations from a frozen ConvNeXt-Tiny encoder. Across both protocols, auxiliary species conditioning statistically significantly degrades ordinal Quadratic Weighted Kappa (QWK) compared to visual ConvNeXt-Tiny representations alone across the evaluated seeds:
- **Historical Split V2:** Config A visual baseline achieves QWK $0.8478 \pm 0.0260$; auxiliary species conditioning drops QWK to $0.8118 \pm 0.0364$ ($\Delta = -0.0360$, 95% CI [$-0.0537, -0.0183$], paired $t(29) = -4.159$, $p = 0.000259$, Wilcoxon signed-rank $W = \min(W^+, W^-) = 71.5$ with $W^+ = 71.5, W^- = 393.5$, exact conditional permutation $p = 0.000514$, SciPy no-ties lookup $p = 0.000555$, asymptotic $p = 0.000928$, Holm-adjusted $p_{\text{adj}} = 0.000514$, Cohen's $d_z = -0.7593$).
- **100% Group Session Holdout:** Config A visual baseline achieves QWK $0.8175 \pm 0.0354$; auxiliary species conditioning drops QWK to $0.7913 \pm 0.0489$ ($\Delta = -0.0262$, 95% CI [$-0.0456, -0.0067$], paired $t(29) = -2.749$, $p = 0.01018$, Wilcoxon signed-rank $W = 125.0$ with $W^+ = 125.0, W^- = 340.0$, exact conditional permutation $p = 0.025888$, SciPy no-ties lookup $p = 0.026229$, asymptotic $p = 0.027025$, Holm-adjusted $p_{\text{adj}} = 0.025888$, Cohen's $d_z = -0.5019$).
Under Holm–Bonferroni multiplicity control ($m=3$), auxiliary species, OOD distance, and joint context conditioning all significantly degrade ordinal grading ($p_{\text{adj}} < 0.05$ across all tests).

Third, evaluating real-image open-set recognition on 5,000 real images of unencountered species (*Cherry, Kiwi, Peach, Pineapple, Plum*), Mahalanobis Distance achieves 84.22% AUROC within clean studio domains, but drops to 18.08% on market-bin images and 18.81% on packing conveyors. In contrast, Free Energy Score maintains robust resilience across domain shifts (92.49% on market bins, 98.89% on packing conveyors) without domain inversion. Finally, confidence-thresholded selective abstention effectively filters sorting errors: for representative Seed 83, abstention at 90% coverage raises accuracy from 93.71% to 99.28% and cuts ordinal cutpoint MAE from 0.0889 to 0.0136 (an exact 84.70% error reduction). Across 30 seeds, mean selective MAE drops from 0.1145 to 0.0217 (81.1% mean reduction), with a 30-seed aggregate mean ECE of 2.11% ± 0.53% (single-seed 83 ECE is 1.47%; 30-seed probability ensemble ECE is 2.51%). Our findings demonstrate that deep visual encoders natively capture species-specific geometry, rendering auxiliary categorical conditioning redundant, and provide an evidence-backed baseline for trustworthy agricultural automation.

---

# Methodology

## 1. Problem Formulation and Ordinal Framework

Let $\mathcal{X} \subset \mathbb{R}^{H \times W \times 3}$ denote the input image space, and let $y \in \mathcal{Y} = \{0, 1, \dots, K-1\}$ denote the ordinal severity grade, where $K=5$ corresponds to Healthy (0), Mild (1), Moderate (2), Severe (3), and Rotten (4). Because misclassifying Grade 0 as Grade 4 carries far greater agricultural and economic penalty than misclassifying Grade 0 as Grade 1, standard multi-class cross-entropy is methodologically inappropriate.

We employ Rank-Consistent Ordinal Regression (CORAL) \cite{cao2020rank}. The ordinal task is decomposed into $K-1$ binary classification sub-tasks sharing a projection vector $\mathbf{w} \in \mathbb{R}^D$ and task-specific cutpoint thresholds $b_k$:

$$h_k(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b_k) \approx P(y > k \mid \mathbf{x}), \quad k \in \{0, \dots, K-2\}$$

## 2. Monotonic CORAL Parameterization

In conventional CORAL implementations, cutpoints $b_k$ are updated independently without constraint. Across random seeds, threshold inversions ($b_{k} > b_{k-1}$) frequently occur, producing invalid cumulative distributions where $P(y > k) > P(y > k-1)$ and yielding negative class probabilities.

To guarantee mathematical consistency, we parameterize cutpoint biases strictly via cumulative softplus decrements:

$$b_0 = \theta_0$$
$$b_k = \theta_0 - \sum_{j=1}^k \text{softplus}(\delta_j), \quad k \in \{1, \dots, K-2\}$$

Because $\text{softplus}(\delta) = \ln(1 + e^\delta) > 0$ strictly for all real $\delta \in \mathbb{R}$, this parameterization ensures:

$$b_0 > b_1 > b_2 > \dots > b_{K-2} \quad \text{unconditionally.}$$

The discrete class probabilities $P(y = k \mid \mathbf{x})$ are reconstructed via:

$$P(y = 0 \mid \mathbf{x}) = 1 - P(y > 0 \mid \mathbf{x})$$
$$P(y = k \mid \mathbf{x}) = P(y > k-1 \mid \mathbf{x}) - P(y > k \mid \mathbf{x}), \quad k \in \{1, \dots, K-2\}$$
$$P(y = K-1 \mid \mathbf{x}) = P(y > K-2 \mid \mathbf{x})$$

This formulation guarantees that $P(y = k \mid \mathbf{x}) \ge 0$ for all $k$ and $\sum_{k=0}^{K-1} P(y = k \mid \mathbf{x}) = 1.0$ within floating-point tolerance. The canonical rank prediction rule is the cumulative threshold count:

$$\hat{y} = \sum_{k=0}^{K-2} \mathbb{I}(P(y > k \mid \mathbf{x}) > 0.5)$$

## 3. Class-Balanced Effective Number Weighting

To combat severe class imbalance ($36,414$ Healthy samples vs $208$ Moderate samples in training) without the extreme gradient instability of raw inverse weighting, we adopt the Effective Number of Samples formulation \cite{cui2019class}:

$$E_{n_k} = \frac{1 - \beta^{n_k}}{1 - \beta}, \quad w_k = \frac{1}{E_{n_k}}, \quad \text{with } \beta = 0.999$$

Weights are normalized such that $\sum_{k=0}^{K-1} w_k = K$. This bounds the maximum-to-minimum weight ratio to $5.32\times$, in contrast to the $175.07\times$ ratio of raw inverse frequency weighting.

## 4. Multi-Modal Context Conditioning Architectures

We formalize four comparative model architectures:
- **Configuration A (Image Only):** Frozen ConvNeXt-Tiny embeddings (768-D) $\to$ 2-layer visual projection (768 $\to$ 512 $\to$ 256) $\to$ fusion MLP (256 $\to$ 128) $\to$ MonotonicCORALHead (128 $\to$ 4 cutpoint logits). Total trainable parameters: **626,180**.
- **Configuration B (Image + Species):** Visual projection (256-D) concatenated with 16-D species embedding from `RobustSpeciesEmbedding` (5 known species + explicit index 5 for unencountered species fallback).
- **Configuration C (Image + OOD):** Visual projection (256-D) concatenated with 2-D Mahalanobis distance representation.
- **Configuration D (Image + Species + OOD):** Full multi-modal concatenation ($256 + 16 + 2 = 274\text{-D}$) $\to$ Fusion MLP $\to$ MonotonicCORALHead.

### Feature Extraction and Frozen Backbone Protocol

All experiments operate on precomputed ConvNeXt-Tiny feature tensors extracted once and cached in `outputs/features/phase4_dataset.pt` (56,036 samples × 768 dimensions). The ConvNeXt-Tiny backbone (~27.8M parameters) is frozen. Only the Monotonic CORAL head and fusion MLP (~626k parameters) receive gradient updates (AdamW, lr=1e-3, weight_decay=1e-4, 15 epochs, batch_size=2048 feature vectors).

## 5. Dataset Partitions and Session Allocation

The dataset comprises 56,036 images partitioned across 80 physical acquisition sessions. We evaluate across two rigorously documented protocols:

1. **Historical Split V2 (`phase50_split_v2_manifest.csv`):**  
   Partitioned into train (39,790), validation (3,413), and test ($N_{\text{test}}=12,833$) splits. While path-level overlap is strictly 0, acquisition session `20200730` was split between train (147 images) and test (119 images). This split is preserved for direct historical comparison and transparency.
2. **100% Group Session Holdout (`session_holdout_manifest.csv`):**  
   To eliminate all turntable/session leakage, all 266 images from session `20200730` are reallocated to the test partition. This results in train (39,643), validation (3,413), and test ($N_{\text{test}}=12,980$) splits, ensuring **100% strict group isolation across all 80 acquisition sessions** (49 train sessions, 10 validation sessions, and 21 test sessions). Zero cross-split session leakage exists on this evaluated partition.

---

# Experimental Results

## 1. Primary Confirmatory Hypothesis: Species Conditioning Degradation

We executed 30 prespecified seeds across all four configurations in both protocols ($N=240$ total independent model runs). To prevent misunderstanding, statistical sample size is denoted by $n_{\text{seeds}} = 30$ (yielding degrees of freedom $\text{df} = 29$), while evaluated test-set image counts are denoted by $N_{\text{test}}$.

### Performance Summary Across Protocols (30 Seeds Mean ± SD)

| Protocol | Config | Architecture | Test QWK | Test MAE | Test Accuracy (%) | Test Macro-F1 (%) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **Historical Split V2** ($N_{\text{test}}=12,833$) | **A** | **Image Only** | **0.8478 ± 0.0260** | **0.1145 ± 0.0178** | **92.11% ± 1.49%** | **49.48% ± 1.82%** |
| Historical Split V2 | **B** | Image + Species | 0.8118 ± 0.0364 | 0.1268 ± 0.0184 | 92.06% ± 1.23% | 49.35% ± 1.91% |
| Historical Split V2 | **C** | Image + OOD | 0.7778 ± 0.0450 | 0.1554 ± 0.0241 | 89.41% ± 1.71% | 42.39% ± 2.14% |
| Historical Split V2 | **D** | Image + Species + OOD | 0.7630 ± 0.0402 | 0.1536 ± 0.0170 | 90.29% ± 1.35% | 44.39% ± 1.87% |
| **100% Session Holdout** ($N_{\text{test}}=12,980$) | **A** | **Image Only** | **0.8175 ± 0.0354** | **0.1342 ± 0.0210** | **90.84% ± 1.62%** | **47.62% ± 1.95%** |
| 100% Session Holdout | **B** | Image + Species | 0.7913 ± 0.0489 | 0.1481 ± 0.0245 | 90.12% ± 1.81% | 46.18% ± 2.08% |
| 100% Session Holdout | **C** | Image + OOD | 0.7650 ± 0.0512 | 0.1712 ± 0.0289 | 88.15% ± 2.05% | 40.85% ± 2.31% |
| 100% Session Holdout | **D** | Image + Species + OOD | 0.7301 ± 0.0581 | 0.1804 ± 0.0261 | 88.92% ± 1.78% | 41.52% ± 2.12% |

### Authoritative Paired Statistical Contrasts vs Config A Baseline

Multiplicity control is applied across the prespecified comparison family $\mathcal{F} = \{B - A, C - A, D - A\}$ ($m=3$) using the Holm–Bonferroni step-down procedure. The Wilcoxon signed-rank statistic is explicitly defined as $W = \min(W^+, W^-)$:

| Protocol | Contrast | $N_{\text{test}}$ | $n_{\text{seeds}}$ | Mean Diff ($\Delta$ QWK) | 95% Conf. Int. | Cohen's $d_z$ | Paired $t(29)$ | Raw $p_t$ | Holm $p_{t,\text{adj}}$ | $W = \min(W^+, W^-)$ | $W^+$ / $W^-$ | Exact Perm $p_w$ | Holm $p_{w,\text{adj}}$ | Asymptotic $p_w$ | Pairs (+/-) |
|:---|:---:|---:|:---:|---:|:---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|:---:|
| **Historical V2** | **B vs A** | 12,833 | 30 | **-0.0360** | [-0.0537, -0.0183] | -0.7593 | -4.1590 | 2.59e-4 | **2.59e-4** | **71.5** | 71.5 / 393.5 | **5.14e-4** | **5.14e-4** | 9.28e-4 | 9 / 21 |
| Historical V2 | **C vs A** | 12,833 | 30 | **-0.0700** | [-0.0902, -0.0498] | -1.2947 | -7.0911 | 8.41e-8 | **1.68e-7** | **16.0** | 16.0 / 449.0 | **3.15e-7** | **6.29e-7** | 8.47e-6 | 2 / 28 |
| Historical V2 | **D vs A** | 12,833 | 30 | **-0.0848** | [-0.1008, -0.0687] | -1.9713 | -10.7971 | 1.13e-11 | **3.38e-11** | **0.0** | 0.0 / 465.0 | **1.86e-9** | **5.59e-9** | 1.73e-6 | 0 / 30 |
| **Session Holdout** | **B vs A** | 12,980 | 30 | **-0.0262** | [-0.0456, -0.0067] | -0.5019 | -2.7490 | 0.01018 | **0.01018** | **125.0** | 125.0 / 340.0 | **0.02589** | **0.02589** | 0.02703 | 10 / 20 |
| Session Holdout | **C vs A** | 12,980 | 30 | **-0.0525** | [-0.0707, -0.0342] | -1.0719 | -5.8715 | 2.27e-6 | **4.53e-6** | **29.0** | 29.0 / 436.0 | **3.24e-6** | **6.48e-6** | 2.84e-5 | 4 / 26 |
| Session Holdout | **D vs A** | 12,980 | 30 | **-0.0874** | [-0.1081, -0.0667] | -1.5743 | -8.6231 | 1.70e-9 | **5.09e-9** | **9.0** | 9.0 / 456.0 | **6.15e-8** | **1.84e-7** | 4.29e-6 | 2 / 28 |

> **Forensic Statistical Resolution:**  
> In Historical B vs A, Seed 3 ($\Delta = +0.0263$) and Seed 73 ($\Delta = -0.0263$) tie at absolute ranks 13 and 14. Standard fractional tie averaging assigns rank $13.5$ to each, yielding positive rank sum $W^+ = 71.5$ and negative rank sum $W^- = 393.5$, with $W = \min(W^+, W^-) = \mathbf{71.5}$.  
> Under the true conditional permutation test conditioned on the observed ranks with ties, exactly $276,128$ sign combinations produce $W^+ \le 71.5$, yielding a two-sided exact conditional permutation $p$-value of **$p = 0.000514$** ($5.143285 \times 10^{-4}$). SciPy 1.17's `method='exact'` evaluated $W=72$ against the theoretical distribution assuming integer ranks without ties ($p = 0.000555$), while the asymptotic normal approximation with tie correction yields **$p = 0.000928$** ($z = -3.3116$). All procedures confirm significant performance degradation ($p < 0.001$). Under the strict session-held-out protocol, degradation persists ($p_{\text{exact\_perm}} = 0.025888$, Holm $p_{\text{adj}} = 0.025888$).

## 2. Real-Image Open-Set Recognition Benchmark

Evaluated against 5,000 real images of unencountered fruit species (*Cherry, Kiwi, Peach, Pineapple, Plum*, $n=1,000$ per species) extracted via the frozen ConvNeXt-Tiny backbone:

| Detector | Feature Space | AUROC (%) | 95% Bootstrap CI | AUPR (OOD=1) (%) | FPR @ 95% TPR (%) | Parameter Fitting Scope |
|:---|:---|---:|:---:|---:|---:|:---|
| **Mahalanobis Distance** | 768-D Embeddings | **84.22%** | [83.7%, 84.8%] | **53.28%** | **38.39%** | Training partition only |
| **Free Energy Score** | Logits ($T=1.0$) | **81.99%** | [81.3%, 82.6%] | **49.96%** | **41.38%** | Validation thresholding at 95% TPR |
| **MSP (Max Softmax)** | Softmax Probs | **74.26%** | [73.6%, 74.9%] | **43.16%** | **47.85%** | Validation thresholding at 95% TPR |

### Domain Breakdown: Mahalanobis Collapse vs Energy Resilience

| In-Distribution Domain | Sample Count | Proportion (%) | Mahalanobis AUROC | Energy Score AUROC | MSP AUROC |
|:---|---:|---:|---:|---:|---:|
| `fruits360` (Studio Turntable) | 10,974 | 85.5% | **93.26%** | 79.99% | 75.21% |
| `real_grade` (Industrial Conveyor) | 67 | 0.5% | **18.81%** | **98.89%** | **96.57%** |
| `real_quality` (Market Bins) | 1,480 | 11.5% | **18.08%** | **92.49%** | **61.74%** |
| `real_severity` (Controlled Lab) | 312 | 2.4% | **94.10%** | **98.67%** | **95.30%** |

> **Scientific Attribution:** Mahalanobis distance collapses to 18.08% on market-bin images because background shift distorts embedding covariance relative to the studio-dominated training distribution. In contrast, Free Energy aggregates unnormalized logits and remains robust across domains ($>92\%$). This is documented as an observed distribution-dependent performance inversion; background causality is not asserted as proven without controlled foreground segmentation ablations.

## 3. Calibration & Selective Prediction

### Reconciled Metrics Across Scopes:
- **Seed 83 Exemplar:** Accuracy = 93.71%, Ordinal Cutpoint MAE = **0.0889**, Categorical Argmax MAE = **0.1234**, ECE (10-bin) = **1.47%**, Brier Score = **0.0857**.
- **30-Seed Aggregate (Mean ± SD):** Accuracy = 92.11% ± 1.49%, Baseline Cutpoint MAE = **0.1145 ± 0.0178**, ECE = **2.11% ± 0.53%**, Brier Score = **0.0940 ± 0.0065**.
- **30-Seed Probability Ensemble:** ECE = **2.51%**, Brier Score = **0.0887**.

### Selective Risk-Coverage Profile (Seed 83):
At 90% coverage (rejecting lowest 10% confidence), selective accuracy rises from 93.71% to **99.28%**, and ordinal cutpoint MAE drops from 0.0889 to **0.0136**—an exact relative error reduction of:

$$\text{Error Reduction} = \frac{0.0889 - 0.0136}{0.0889} \times 100\% = \mathbf{84.70\%}$$

Across all 30 seeds, 90% coverage yields mean selective accuracy of **98.91% ± 0.23%** and mean selective MAE of **0.0217 ± 0.0053** (an **81.1%** mean error reduction).

## 4. Computational Efficiency and Parameter Census

Forensic parameter census from `model_A_seed83_best.pth`:
- **Backbone:** ConvNeXt-Tiny (27,820,128 parameters, frozen).
- **Trainable Head:** Exactly **626,180 parameters** (`vis_proj`: 526,080; `fusion_mlp`: 99,968; `head`: 132).
- **Head-Only Throughput:** >12,000 FPS at batch size 16 and >300,000 FPS at batch size 2,048 on an NVIDIA RTX 3050 Laptop GPU.
- **Full End-to-End Pipeline (Image $\to$ Backbone $\to$ Head on RTX 3050):**
  - **FP16:** 49.0 FPS (latency 20.43 ms) at BS=1; 88.6 FPS (11.29 ms/sample) at BS=16; **90.4 FPS** (11.07 ms/sample) at BS=32.
  - **FP32:** 27.0 FPS (37.10 ms) at BS=1; 39.6 FPS (25.27 ms/sample) at BS=16; **40.1 FPS** (24.92 ms/sample) at BS=32.
Both FP16 throughputs exceed mechanical conveyor sorting line speeds cited in post-harvest automation literature (typically 10–30 fruits/sec).

---

# Limitations and Threats to Validity

1. **Synthetic Laboratory Domain Domination:**  
   The evaluation test set contains 85.5% images from `fruits360`. While high overall accuracy (92.11%) is achieved, accuracy on genuine market bins (`real_quality`) drops to 39.86%. Models trained on clean turntable fruit silhouettes do not generalize zero-shot to complex commercial bins without domain adaptation.
2. **Observed OOD Detector Inversion:**  
   Mahalanobis distance AUROC drops severely from 93.26% on studio backgrounds to 18.08% on market-bin images, while Energy score remains robust (92.49% on `real_quality`). Causal attribution of the Mahalanobis failure specifically to background versus foreground features remains a hypothesis, requiring foreground-segmented validation in future work.
3. **Minority Grade Imbalance:**  
   Grade 2 (Moderate Decay) represents only 0.52% of training data ($n=208$) and 2.93% of session-held-out test data ($n=380$). While Configuration B improves Grade 2 recall from 7.4% to 21.5%, overall minority grade coverage remains sparse across species.
4. **Hardware and Deployment Bounds:**  
   The reported throughput (49.0–90.4 FPS in FP16 on RTX 3050) measures forward PyTorch execution with CUDA synchronization. Translating this computational throughput to a commercial conveyor line requires integrating camera trigger acquisition, lighting stabilization, real-time object detection/cropping, and physical robotic pneumatic ejection.

---

# Reproducibility Checklist

- [x] **Frozen Split Manifests:** Historical split V2 and 100% session holdout manifests persisted with bitwise SHA-256 hashes.
- [x] **Deterministic Seeds:** Exact schedule of 30 prespecified seeds documented across all 240 runs.
- [x] **Zero Test-Set Hyperparameter Tuning:** Checkpoint selection performed strictly on validation QWK.
- [x] **100% Artifact Persistence:** All 240 model checkpoints (`.pth`) and raw per-sample prediction tensors (`.pt`) saved in `outputs/`.
- [x] **Automated Test Suite:** 28 unit, property, and regression tests pass with 100% success (`tests/`) in 115.36s.
- [x] **Empirical Hardware Benchmarks:** Synchronized CUDA benchmarks on NVIDIA RTX 3050 Laptop GPU.
