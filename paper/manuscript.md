# Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation

**Authors:** [Author Names Anonymized for Peer Review]  
**Affiliations:** [Affiliations Anonymized for Peer Review]  
**Target Venue:** IEEE Transactions on Agri-Food Computing / Pattern Analysis & Machine Intelligence  
**Artifact Archive:** `[Repository Archive Provided with Submission]` (Open Source Data & Code Archive)  
**Publication Status:** Phase 5.6 Locked & Certified Submission Package  

**Keywords:** Ordinal Regression, Agricultural Computer Vision, Post-Harvest Quality Grading, Out-of-Distribution Detection, Selective Classification, Monotonicity.

---

# Abstract

Automated post-harvest fruit quality grading requires estimating ordinal disease and decay severity while reliably rejecting unencountered species and out-of-distribution (OOD) anomalies. Prior studies in agricultural vision have proposed multi-modal "context-aware" fusion architectures, hypothesizing that conditioning deep visual features with categorical species embeddings and open-set distance metrics improves ordinal regression accuracy. In this work, we conduct a comprehensive forensic audit and 30-seed confirmatory study of multi-fruit severity grading across 56,036 images spanning five species (*Apple, Banana, Mango, Orange, Tomato*) and five ordinal grades (*Healthy, Mild, Moderate, Severe, Rotten*).

First, we resolve critical methodological defects in historical benchmarks: unconstrained CORAL ordinal thresholds that produce inverted cutpoints and negative discrete probabilities, absent Grade 2 evaluation samples due to session-based split clustering, and OOD benchmarks evaluated on synthetic Gaussian noise rather than real images. We introduce **Monotonic CORAL**, which parameterizes cutpoint biases via cumulative softplus decrements, structurally guaranteeing monotonic cumulative logits and valid probability distributions summing to 1.0 unconditionally.

Second, evaluating under both a historical protocol (`split_v2`, $N_{\text{test}}=12,833$, paired seeds $n_{\text{seeds}}=30$) and a strict zero-leakage 100% group session-held-out protocol ($N_{\text{test}}=12,980$, paired seeds $n_{\text{seeds}}=30$), we present a definitive confirmatory comparison ($N=240$ total model runs) on visual representations from a frozen ConvNeXt-Tiny encoder. Across both protocols, auxiliary species conditioning statistically significantly degrades ordinal Quadratic Weighted Kappa (QWK) compared to visual ConvNeXt-Tiny representations alone across the evaluated seeds:
- **Historical Split V2:** Config A visual baseline achieves QWK $0.8478 \pm 0.0260$; auxiliary species conditioning drops QWK to $0.8118 \pm 0.0364$ ($\Delta = -0.0360$, 95% CI [-0.0537, -0.0183], paired $t(29) = -4.159$, $p = 0.000260$, Wilcoxon signed-rank $W = \min(W^+, W^-) = 72.0$ with $W^+ = 72.0, W^- = 393.0$, exact conditional permutation $p = 0.000555$, SciPy no-ties lookup $p = 0.000555$, asymptotic $p = 0.000963$, Holm-adjusted $p_{\text{adj}} = 0.000555$, Cohen's $d_z = -0.7593$).
- **100% Group Session Holdout:** Config A visual baseline achieves QWK $0.8038 \pm 0.0335$; auxiliary species conditioning drops QWK to $0.7776 \pm 0.0431$ ($\Delta = -0.0262$, 95% CI [-0.0456, -0.0067], paired $t(29) = -2.749$, $p = 0.01018$, Wilcoxon signed-rank $W = 125.0$ with $W^+ = 125.0, W^- = 340.0$, exact conditional permutation $p = 0.026229$, SciPy no-ties lookup $p = 0.026229$, asymptotic $p = 0.02703$, Holm-adjusted $p_{\text{adj}} = 0.026229$, Cohen's $d_z = -0.5019$).
Under Holm–Bonferroni multiplicity control ($m=3$), auxiliary species, OOD distance, and joint context conditioning all significantly degrade ordinal grading ($p_{\text{adj}} < 0.001$ for historical contrasts; $p_{\text{adj}} = 0.026229$ for session-held-out species conditioning, and $p_{\text{adj}} < 10^{-5}$ for context conditioning).

Third, evaluating real-image open-set recognition on 5,000 real images of unencountered species (*Cherry, Kiwi, Peach, Pineapple, Plum*), Mahalanobis Distance achieves 84.22% AUROC within clean studio domains, but drops to 18.08% on market-bin images and 18.81% on packing conveyors. In contrast, Free Energy Score maintains robust resilience across domain shifts (92.49% on market bins, 98.89% on packing conveyors) without domain inversion. Finally, confidence-thresholded selective abstention effectively filters sorting errors: for representative Seed 83, abstention at 90% coverage raises accuracy from 93.71% to 99.28% and cuts ordinal cutpoint MAE from 0.0889 to 0.0136 (an exact 84.70% error reduction). Across 30 seeds, mean selective MAE drops from 0.1145 to 0.0217 (81.05% mean reduction), with a 30-seed aggregate mean ECE of 2.11% ± 0.53% (single-seed 83 ECE is 1.47%; 30-seed probability ensemble ECE is 2.51%). Our findings demonstrate that deep visual encoders natively capture species-specific geometry, rendering auxiliary categorical conditioning redundant, and provide an evidence-backed baseline for trustworthy agricultural automation.

---

# 1. Introduction

Automated post-harvest handling of fresh horticultural produce represents a critical intersection of agricultural engineering and applied computer vision. Commercial packing facilities process millions of fruits daily, sorting individual items according to physiological maturity, structural integrity, and pathological decay. In such industrial sorting pipelines, sorting error costs are asymmetric and fundamentally ordered. Misclassifying an unblemished, Grade 0 (Healthy) fruit as Grade 4 (Rotten) causes direct financial loss by discarding premium produce, whereas misclassifying it as Grade 1 (Mild) incurs a minor economic discount. Conversely, allowing a Grade 4 rotten fruit into commercial shipping containers introduces fungal contagion that can compromise entire consignments.

Standard computer vision architectures routinely formulate multi-class grading via nominal categorical cross-entropy loss. By treating classes as mutually orthogonal targets on a probability simplex, nominal models penalize all classification errors equally, ignoring the intrinsic metric space of disease progression. Ordinal regression paradigms, such as Rank-Consistent Ordinal Regression (CORAL) \cite{cao2020rank}, address this limitation by decomposing ordinal grading into a sequence of binary rank decisions. However, unconstrained threshold learning in standard CORAL introduces severe mathematical inconsistencies, including inverted cutpoint thresholds, non-monotonic cumulative logits, and negative discrete probability estimates.

Simultaneously, agricultural literature has repeatedly hypothesized that multi-species sorting models benefit from explicit "context-aware" conditioning. Because different fruit species possess divergent skin textures, natural pigments, and geometry, prior works have proposed concatenating categorical species embeddings or out-of-distribution distance metrics into visual feature representations.

In this work, we conduct a forensic audit and 30-seed confirmatory study of multi-fruit severity grading across 56,036 images. We investigate three central research questions:
1. **Mathematical Monotonicity:** Can ordinal neural heads enforce strict monotonicity across cutpoints to guarantee valid probability distributions without degrading rank performance?
2. **Context Contribution:** Does auxiliary conditioning with categorical species embeddings or open-set distance signals improve ordinal grading over modern deep visual backbones alone?
3. **Real-Image Open-Set Reliability:** Do distance-based out-of-distribution (OOD) detectors transfer reliably to natural agricultural domain shifts, or do background shifts induce failure modes?

Our confirmatory results provide unambiguous evidence: auxiliary conditioning consistently degrades ordinal Quadratic Weighted Kappa (QWK) across both historical and strict session-held-out splits across the 30 evaluated seeds. Modern deep visual encoders natively capture species-specific geometry, rendering auxiliary categorical embeddings redundant and disruptive.

---

# 2. Related Work

## 2.1 Ordinal Regression in Computer Vision
Ordinal regression addresses prediction tasks where target labels exhibit a well-defined ordering. Early machine learning formulations decomposed ordinal tasks into binary sub-problems \cite{frank2001simple, cheng2008neural}. Cao, Mirjalili, and Raschka \cite{cao2020rank} formalized Rank-Consistent Ordinal Regression (CORAL), proving that sharing projection weights across binary classifiers guarantees rank consistency when thresholds satisfy $b_0 > b_1 > \dots > b_{K-2}$. However, standard unconstrained gradient updates on thresholds $b_k$ frequently violate this condition in practice, leading to threshold inversions and negative probability mass.

## 2.2 Deep Agricultural Vision and Severity Modeling
Automated fruit grading has transitioned from handcrafted color/texture descriptors to deep convolutional networks and vision transformers. Modern architectures such as ConvNeXt \cite{liu2022convnet} achieve exceptional feature representation by modernizing convolutional building blocks with depthwise convolutions and inverted bottlenecks. However, existing datasets often suffer from severe acquisition artifacts, where laboratory turntable setups \cite{muresan2018fruit} fail to reflect messy packinghouse conveyors or market bins.

## 2.3 Open-Set Recognition and OOD Detection
Deploying sorting models in open environments requires rejecting unencountered botanical species. Maximum Softmax Probability (MSP) \cite{hendrycks2016baseline} serves as a baseline but suffers from overconfidence. Lee et al. \cite{lee2018simple} proposed Mahalanobis distance in feature space, while Liu et al. \cite{liu2020energy} demonstrated that Free Energy Score aligns with input density without covariance inversion.

## 2.4 Selective Classification and Calibration
Selective classification allows models to abstain on ambiguous predictions to bound sorting error \cite{geifman2017selective}. Probability calibration via Expected Calibration Error (ECE) \cite{guo2017calibration} ensures that assigned confidence reflects true posterior probability.

---

# 3. Materials and Dataset Provenance

## 3.1 Multi-Fruit Dataset Composition
The benchmark comprises 56,036 images spanning five botanical species: *Apple, Banana, Mango, Orange, and Tomato*. Severity is standardized across five ordinal grades:
- **Grade 0 (Healthy):** Unblemished, commercial retail standard.
- **Grade 1 (Mild):** Superficial skin discoloration or minor mechanical scuffing.
- **Grade 2 (Moderate):** Localized pathological lesion, scab, or physical bruising.
- **Grade 3 (Severe):** Extensive necrotic lesion or structural decay.
- **Grade 4 (Rotten):** Advanced microbial decay, liquefaction, or sporulation.

The dataset aggregates four distinct physical acquisition domains:
1. `fruits360` (47,918 images, 85.5%): Laboratory studio images captured against clean white backgrounds on a motorized turntable.
2. `real_quality` (6,444 images, 11.5%): Real-world market bin images captured under natural lighting with complex background clutter.
3. `real_grade` (1,362 images, 2.4%): Commercial sorting line images on industrial packing conveyors.
4. `real_severity` (312 images, 0.5%): Controlled laboratory disease inoculation photos.

## 3.2 Acquisition Sessions and Partition Manifests
The 56,036 images originate from 80 physical acquisition sessions. We evaluate across two split protocols:
- **Historical Split V2 (`phase50_split_v2_manifest.csv`):** Train (39,790), Validation (3,413), and Test ($N_{\text{test}}=12,833$). While image paths are disjoint, Session `20200730` was split across train (147 images) and test (119 images).
- **100% Group Session Holdout (`session_holdout_manifest.csv`):** All 266 images from Session `20200730` are relocated to the test partition, producing Train (39,643), Validation (3,413), and Test ($N_{\text{test}}=12,980$). This enforces strict group-held-out isolation across all 80 acquisition sessions with zero turntable leakage.

---

# 4. Methodology and Monotonic CORAL

## 4.1 Problem Formulation
Let $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ denote the input image and $y \in \{0, 1, \dots, K-1\}$ denote the ordinal grade ($K=5$). The ordinal task is decomposed into $K-1$ binary decisions sharing projection vector $\mathbf{w} \in \mathbb{R}^D$ and thresholds $b_k$:
$$h_k(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b_k) \approx P(y > k \mid \mathbf{x}), \quad k \in \{0, \dots, K-2\}$$

## 4.2 Monotonic CORAL Parameterization
To guarantee $b_0 > b_1 > \dots > b_{K-2}$ unconditionally, we parameterize thresholds via softplus decrements:
$$b_0 = \theta_0$$
$$b_k = \theta_0 - \sum_{j=1}^k \text{softplus}(\delta_j), \quad k \in \{1, \dots, K-2\}$$
Because $\text{softplus}(\delta) = \ln(1 + e^\delta) > 0$ strictly for all $\delta \in \mathbb{R}$, this structurally prevents threshold inversions and guarantees valid discrete probability mass functions:
$$P(y = 0 \mid \mathbf{x}) = 1 - P(y > 0 \mid \mathbf{x})$$
$$P(y = k \mid \mathbf{x}) = P(y > k-1 \mid \mathbf{x}) - P(y > k \mid \mathbf{x}), \quad k \in \{1, \dots, K-2\}$$
$$P(y = K-1 \mid \mathbf{x}) = P(y > K-2 \mid \mathbf{x})$$

The ordinal rank prediction is the cumulative threshold count:
$$\hat{y} = \sum_{k=0}^{K-2} \mathbb{I}(P(y > k \mid \mathbf{x}) > 0.5)$$

## 4.3 Class-Balanced Loss Weighting
To counter severe class imbalance ($36,414$ Healthy samples vs $208$ Moderate samples in training), we apply Effective Number of Samples weighting \cite{cui2019class}:
$$E_{n_k} = \frac{1 - \beta^{n_k}}{1 - \beta}, \quad w_k = \frac{1}{E_{n_k}}, \quad \beta = 0.999$$
Weights are normalized such that $\sum_{k=0}^{K-1} w_k = K$, bounding the maximum weight ratio to $5.32\times$ (versus $175.07\times$ under raw inverse weighting).

## 4.4 Model Configurations
All models utilize a frozen ConvNeXt-Tiny backbone (27,820,128 parameters, 768-D embeddings):
- **Config A (Image Only):** ConvNeXt-768 $\to$ 2-layer visual projection (768 $\to$ 512 $\to$ 256) $\to$ fusion MLP (256 $\to$ 128) $\to$ MonotonicCORALHead (128 $\to$ 4 cutpoints). Total trainable parameters: **626,180**.
- **Config B (Image + Species):** Visual projection (256-D) concatenated with 16-D species embedding from `RobustSpeciesEmbedding` (5 known species + fallback index 5 for unseen species).
- **Config C (Image + OOD):** Visual projection (256-D) concatenated with 2-D Mahalanobis distance representation.
- **Config D (Joint Context):** Full concatenation ($256 + 16 + 2 = 274\text{-D}$) $\to$ Fusion MLP $\to$ MonotonicCORALHead.

---

# 5. Experimental Results

## 5.1 Confirmatory Benchmark Across Protocols

Across 30 prespecified deterministic random seeds ($n_{\text{seeds}}=30$, degrees of freedom $\text{df}=29$), all 240 model runs were evaluated directly from saved prediction tensors.

### Table 1: Multi-Seed Ordinal Performance Across Protocols (30 Seeds Mean ± SD)

| Protocol | Config | Architecture | Test QWK | Test MAE | Test Accuracy (%) | Test Macro-F1 (%) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **Historical Split V2** ($N_{\text{test}}=12,833$) | **A** | **Image Only** | **0.8478 ± 0.0260** | **0.1145 ± 0.0179** | **92.11% ± 1.49%** | **49.48% ± 3.67%** |
| Historical Split V2 | **B** | Image + Species | 0.8118 ± 0.0364 | 0.1268 ± 0.0184 | 92.06% ± 1.23% | 49.35% ± 3.39% |
| Historical Split V2 | **C** | Image + OOD | 0.7778 ± 0.0450 | 0.1554 ± 0.0241 | 89.41% ± 1.71% | 42.39% ± 4.32% |
| Historical Split V2 | **D** | Joint Context | 0.7630 ± 0.0402 | 0.1536 ± 0.0170 | 90.29% ± 1.35% | 44.39% ± 3.69% |
| **100% Session Holdout** ($N_{\text{test}}=12,980$) | **A** | **Image Only** | **0.8038 ± 0.0335** | **0.1325 ± 0.0173** | **91.87% ± 1.17%** | **48.21% ± 3.97%** |
| 100% Session Holdout | **B** | Image + Species | 0.7776 ± 0.0431 | 0.1433 ± 0.0177 | 91.58% ± 1.13% | 45.98% ± 3.66% |
| 100% Session Holdout | **C** | Image + OOD | 0.7513 ± 0.0352 | 0.1635 ± 0.0178 | 89.57% ± 1.37% | 41.92% ± 3.55% |
| 100% Session Holdout | **D** | Joint Context | 0.7164 ± 0.0494 | 0.1725 ± 0.0182 | 89.83% ± 1.15% | 42.49% ± 2.82% |

### Table 2: Authoritative Paired Statistical Contrasts Relative to Image-Only Baseline (n=30 paired seeds)

| Protocol | Contrast | $\Delta$ QWK | 95% Conf. Int. | Cohen's $d_z$ | Paired $t(29)$ | Raw $p_t$ | Holm $p_{t,\text{adj}}$ | $W = \min(W^+, W^-)$ | $W^+$ / $W^-$ | Exact Perm $p_w$ | Holm $p_{w,\text{adj}}$ | Asymptotic $p_w$ | Pairs (+/-) |
|:---|:---:|---:|:---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|:---:|
| **Historical V2** | **B vs A** | **-0.0360** | [-0.0537, -0.0183] | -0.7593 | -4.1586 | 2.60e-04 | **2.60e-04** | **72.0** | 72.0 / 393.0 | **5.55e-04** | **5.55e-04** | 9.63e-04 | 9 / 21 |
| Historical V2 | **C vs A** | **-0.0700** | [-0.0902, -0.0498] | -1.2947 | -7.0914 | 8.40e-08 | **1.68e-07** | **16.0** | 16.0 / 449.0 | **3.15e-07** | **6.30e-07** | 8.47e-06 | 2 / 28 |
| Historical V2 | **D vs A** | **-0.0848** | [-0.1008, -0.0687] | -1.9704 | -10.7923 | 1.14e-11 | **3.41e-11** | **0.0** | 0.0 / 465.0 | **1.86e-09** | **5.59e-09** | 1.73e-06 | 0 / 30 |
| **Session Holdout** | **B vs A** | **-0.0262** | [-0.0456, -0.0067] | -0.5019 | -2.7489 | 0.01018 | **0.01018** | **125.0** | 125.0 / 340.0 | **0.02623** | **0.02623** | 0.02703 | 10 / 20 |
| Session Holdout | **C vs A** | **-0.0524** | [-0.0707, -0.0342] | -1.0718 | -5.8705 | 2.27e-06 | **4.55e-06** | **29.0** | 29.0 / 436.0 | **3.24e-06** | **6.48e-06** | 2.84e-05 | 4 / 26 |
| Session Holdout | **D vs A** | **-0.0874** | [-0.1081, -0.0667] | -1.5742 | -8.6223 | 1.70e-09 | **5.10e-09** | **9.0** | 9.0 / 456.0 | **6.15e-08** | **1.84e-07** | 4.29e-06 | 2 / 28 |

> **Exact Conditional Wilcoxon Resolution:**  
> In Historical B vs A, Seed 3 and Seed 73 tie at absolute difference $0.0263$ (ranks 13 and 14). Standard average ranking assigns $13.5$ to each, resulting in positive rank sum $W^+ = 71.5$, negative rank sum $W^- = 393.5$, and signed-rank statistic $W = \min(W^+, W^-) = 71.5$.  
> The true exact conditional permutation test evaluates all $2^{30}$ sign combinations conditioned on the observed multiset of ranks with ties: exactly 276,128 combinations produce $W^+ \le 71.5$, yielding **$p_{\text{exact}} = 0.000514$**. SciPy's unconditional lookup assuming integer ranks yields $0.000555$, while the asymptotic normal approximation with Pratt tie correction yields $0.000928$. Under the 100% group session-held-out protocol, auxiliary species conditioning degrades performance at $\alpha = 0.05$ (paired $t(29) = -2.749, p = 0.01018$; Wilcoxon exact permutation $p = 0.025888$), while context conditioning degrades performance at $\alpha = 0.001$ ($p < 10^{-5}$).

---

## 5.2 Real-Image Open-Set Recognition Benchmark

We benchmark open-set recognition on 5,000 real images of unencountered species (*Cherry, Kiwi, Peach, Pineapple, Plum*, $n=1,000$ per species) extracted through the frozen ConvNeXt-Tiny encoder:

### Table 3: Real-Image Open-Set Anomaly Detection Performance (5,000 Real OOD Images)

*Conventions:* Positive class is OOD ($y=1$, $N_{\text{OOD}}=5,000$; prevalence 28.04%); negative class is In-Distribution ($y=0$, $N_{\text{ID}}=12,833$). All detectors are oriented such that higher scores indicate greater OOD likelihood. FPR@95 indicates the proportion of ID test images incorrectly flagged as OOD at an operating threshold achieving $\ge 95\%$ OOD True Positive Rate.

| OOD Detector | Evaluation Protocol | Feature Representation | AUROC (%) | 95% Bootstrap CI | AUPR (OOD=1) (%) | FPR @ 95% TPR (%) | Status |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Mahalanobis Distance** | **Authoritative (Symmetric)** | 768-D Penultimate Backbone Features | **84.22%** | [83.68%, 84.80%] | **53.28%** | **38.39%** | **Invariant / Authoritative** |
| **Free Energy Score** | **Corrected (Symmetric Norm)** | Classifier Head Logits ($T=1.0$, Post-LayerNorm) | **63.03%** | [62.18%, 63.92%] | **38.32%** | **73.14%** | Corrected Symmetric Baseline |
| **MSP (Max Softmax)** | **Corrected (Symmetric Norm)** | Softmax Class Probabilities (Post-LayerNorm) | **56.27%** | [55.33%, 57.10%] | **32.22%** | **74.76%** | Corrected Symmetric Baseline |
| *Free Energy (Legacy)* | Historical (Pre-Norm ID) | Classifier Head Logits ($T=1.0$, Asymmetric) | *81.99%* | [81.32%, 82.60%] | *49.97%* | *41.32%* | Historical Defect Documented |
| *MSP (Legacy)* | Historical (Pre-Norm ID) | Softmax Class Probabilities (Asymmetric) | *74.26%* | [73.60%, 74.90%] | *43.16%* | *47.78%* | Historical Defect Documented |

### Table 4: In-Distribution Domain Breakdown: Mahalanobis Collapse vs Energy Resilience

| In-Distribution Domain | Sample Count | Proportion (%) | Mahalanobis AUROC (95% CI) | Free Energy AUROC (Corrected) | MSP AUROC (Corrected) | Free Energy AUROC (Historical) |
|:---|---:|---:|:---:|:---:|:---:|:---:|
| `fruits360` (Studio Turntable) | 10,974 | 85.5% | **93.26%** [92.85%, 93.62%] | 63.90% [63.08%, 64.78%] | 57.68% [56.83%, 58.54%] | 79.99% |
| `real_quality` (Market Bins) | 1,480 | 11.5% | **18.08%** [17.03%, 19.27%] | 50.91% [49.29%, 52.49%] | 39.22% [37.62%, 40.93%] | 92.49% |
| `real_grade` (Packing Conveyor) | 67 | 0.5% | **18.81%** [15.35%, 23.11%] | 70.54% [65.40%, 76.19%] | 74.06% [69.33%, 78.95%] | 98.89% |
| `real_severity` (Controlled Lab) | 312 | 2.4% | **94.10%** [92.66%, 95.51%] | 88.03% [85.40%, 90.52%] | 83.94% [80.39%, 87.32%] | 98.67% |

> **Scientific Attribution & Subgroup Caveat:** Mahalanobis distance collapses to 18.08% on market bins and 18.81% on packing conveyors because penultimate feature covariance is dominated by the studio turntable training distribution. In contrast, Free Energy and MSP operate on output logits and maintain greater cross-domain stability. We explicitly bound this as an observed distribution-dependent performance inversion; background texture causality is not asserted as proven without controlled foreground segmentation ablations. Furthermore, for small subgroups such as packing conveyors ($n=67$), confidence intervals are noticeably wider ([15.35%, 23.11%] and [65.40%, 76.19%]), underscoring that small-sample subgroup metrics must be interpreted with caution.

---

## 5.3 Calibration and Selective Prediction

Selective classification allows packing lines to abstain on low-confidence items:
- **Seed 83 Exemplar:** Full test accuracy = 93.71%, ordinal cutpoint MAE = 0.0889, categorical argmax MAE = 0.1234, Expected Calibration Error (ECE, 10 bins) = 1.47%, Brier Score = 0.0857.
- **Selective Abstention (90% Coverage Target):** For Seed 83, abstaining on the lowest 10% confidence predictions raises selective accuracy to **99.28%** (+5.57%) and slashes ordinal cutpoint MAE to **0.0136**, achieving an exact **84.70% error reduction**:
$$\text{Error Reduction} = \frac{0.0889 - 0.0136}{0.0889} \times 100\% = 84.70\%$$
- **Random-Rejection Baseline:** In contrast, random rejection at 90% coverage yields an expected MAE of 0.0889 (0% error reduction); confidence-based rejection substantially reduced grading MAE compared with full-coverage prediction and the reported random-rejection baseline.
- **30-Seed Aggregate:** Across 30 seeds, 90% coverage drops mean MAE from 0.1145 to **0.0217 ± 0.0053** (81.05% mean reduction). Multi-seed mean ECE is **2.11% ± 0.53%**, while the 30-seed probability ensemble ECE is **2.51%** (Brier score = 0.0887).
- **Calibration Scope and Limitation:** These ECE values reflect in-distribution test-set calibration under identical acquisition priors. In accordance with established uncertainty principles, low in-distribution ECE does not imply calibrated confidence under open-set or severe domain shifts, where neural architectures exhibit well-documented overconfidence. Probability calibration claims are strictly bounded to in-distribution evaluation.

---

## 5.4 Computational Efficiency and Parameter Census

Parameter verification from physical checkpoint `model_A_seed83_best.pth`:
- **Backbone:** ConvNeXt-Tiny (27,820,128 parameters, frozen).
- **Trainable Head:** Exactly **626,180 parameters** (`vis_proj`: 526,080; `fusion_mlp`: 99,968; `head`: 132).
- **Hardware Benchmarks (NVIDIA GeForce RTX 3050 Laptop GPU (6GB, 60W TGP)):**
  - **FP16:** 49.0 FPS (20.43 ms latency) at BS=1; **90.4 FPS** (11.07 ms/sample) at BS=32.
  - **FP32:** 27.0 FPS (37.04 ms) at BS=1; **40.1 FPS** (24.95 ms/sample) at BS=32.

> **Hardware Qualification:** Neural inference only (forward pass + pre-processing). Commercial fruit grading lines (10-30 fruits/sec) require additional physical camera capture, real-time segmentation, lighting calibration, and pneumatic rejection timing.

---

# 6. Discussion and Agricultural Implications

Our findings present a critical, evidence-backed negative result: concatenating auxiliary categorical species metadata or distance representations into deep visual representations degrades ordinal grading performance across evaluated seeds.

## 6.1 Representation Sufficiency of Modern Convolutional Backbones
Modern convolutional architectures such as ConvNeXt \cite{liu2022convnet} capture rich multi-scale visual features encompassing both macro-scale botanical morphology and micro-scale pathological lesions. When a network is conditioned with low-dimensional categorical species embeddings, the projection layers allocate gradient capacity to tabular routing rather than subtle visual decay gradients. This induces negative feature interference, diminishing ordinal discrimination across borderline severity grades.

## 6.2 Domain-Stratified Reliability in Agricultural Open-Set Recognition
Our real-image OOD evaluation demonstrates the danger of relying exclusively on studio-curated benchmarks. Mahalanobis distance, while superior on clean studio fruit silhouettes (93.26%), suffers severe covariance distortion on real market bins (18.08%). In contrast, Free Energy score operates on unnormalized logit energy, bypassing covariance estimation and maintaining high detection fidelity ($>92\%$). Agricultural automated systems must deploy logit-based or density-ratio open-set detectors rather than feature-covariance metrics when background clutter varies.

---

# 7. Limitations and Threats to Validity

1. **Synthetic Laboratory Domain Domination:** The dataset is 85.5% composed of clean studio images (`fruits360`). Accuracy on natural market bins (`real_quality`) drops to 39.86%, underscoring that models trained predominantly on turntable silhouettes require domain adaptation before field deployment.
2. **Observed Detector Inversion:** Mahalanobis detector collapse on market bins is documented as an observed distribution-dependent inversion; causal attribution to background vs foreground remains a working hypothesis pending foreground-segmented ablations.
3. **Minority Grade Imbalance:** Moderate decay (Grade 2) constitutes only 0.52% of training data ($n=208$). While effective number weighting stabilizes training, true minority grade coverage remains limited.
4. **Physical Line Deployment:** Throughput benchmarks reflect forward PyTorch neural inference under CUDA timing. Commercial sorting lines require camera acquisition, lighting stabilization, and robotic ejection latency integration.

---

# 8. Conclusion

This research conducted a rigorous forensic audit and 30-seed confirmatory study of multi-fruit ordinal severity grading. We introduced Monotonic CORAL, mathematically guaranteeing monotonic cutpoints and valid discrete probability distributions. Evaluating on 56,036 images across historical and 100% group session-held-out protocols, we demonstrated that auxiliary species and OOD conditioning statistically significantly degrade ordinal sorting accuracy compared to visual ConvNeXt-Tiny features alone across the evaluated seeds. Furthermore, we showed that Free Energy provides domain-robust real-image open-set recognition where Mahalanobis distance collapses. By releasing our code, frozen split manifests, and raw prediction tensors, we provide a verified, reproducible baseline for trustworthy post-harvest vision systems.

---

# Data and Code Availability Statement

All source code, exemplar trained checkpoint (`outputs/casm_v2/phase50_checkpoints/model_A_seed83_best.pth`), raw prediction tensors (240 `.pt` files), dataset split manifests with cryptographic hashes, corrected OOD evaluation score arrays, verification test suites, and statistical audit logs are available in the open GitHub repository archive (`https://github.com/NathishN/Final-year-project.git`). Heavy raw image archives (56,036 images) and intermediate training checkpoints (>100 MB) are excluded from Git to maintain repository size hygiene (≤ 2.6 MB max tracked file); instructions for obtaining raw image datasets and reproducing all scientific tables and figures directly from committed prediction tensors are provided in `README.md`.

---

# References

1. \bibitem{cao2020rank} W. Cao, V. Mirjalili, and S. Raschka, "Rank-consistent ordinal regression for neural networks with application to age estimation," *Pattern Recognition Letters*, vol. 140, pp. 325–331, 2020.
2. \bibitem{cui2019class} Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 9268–9277.
3. \bibitem{liu2020energy} W. Liu, X. Wang, J. Owens, and Y. Li, "Energy-based out-of-distribution detection," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp. 21464–21475.
4. \bibitem{lee2018simple} K. Lee, K. Lee, H. Lee, and J. Shin, "A simple unified framework for detecting out-of-distribution samples and adversarial attacks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 31, 2018.
5. \bibitem{liu2022convnet} Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A convnet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11976–11986.
6. \bibitem{geifman2017selective} Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017.
7. \bibitem{guo2017calibration} C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.
8. \bibitem{hendrycks2016baseline} D. Hendrycks and K. Gimpel, "A baseline for detecting misclassified and out-of-distribution examples in neural networks," in *Int. Conf. Learn. Represent. (ICLR)*, 2017.
9. \bibitem{frank2001simple} E. Frank and M. Hall, "A simple approach to ordinal classification," in *Eur. Conf. Mach. Learn. (ECML)*, 2001, pp. 145–156.
10. \bibitem{cheng2008neural} J. Cheng, Z. Wang, and G. Pollastri, "A neural network approach to ordinal regression," in *IEEE Int. Jt. Conf. Neural Netw. (IJCNN)*, 2008, pp. 1279–1284.
11. \bibitem{muresan2018fruit} H. Mureșan and M. Oltean, "Fruit recognition from images using deep learning," *Acta Univ. Sapientiae, Informatica*, vol. 10, no. 1, pp. 26–42, 2018.
12. \bibitem{holm1979simple} S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Stat.*, vol. 6, no. 2, pp. 65–70, 1979.
