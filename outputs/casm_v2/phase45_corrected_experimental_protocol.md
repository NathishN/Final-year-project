# Phase 4.5 — Corrected Experimental Protocol for Publication-Grade Confirmatory Study

**Authors & Methodologists:** Senior ML Research Engineer, Computer Vision Scientist, Statistical Auditor, IEEE Transactions Reviewer  
**Date:** October 2026  
**Repository:** `C:\fruit_dataset`  
**Deliverable File:** `outputs/casm_v2/phase45_corrected_experimental_protocol.md`

---

## 1. Executive Summary and Protocol Rationale

Prior experimental phases (Phase 4.0 through 4.4) suffered from five methodological vulnerabilities that prevent publication in premier IEEE Transactions venues:
1. **Ghost Class in Test Split:** The historical test set contained zero samples of Grade 2 ($N_2 = 0$), invalidating 5-class Macro-F1 and 5-class ordinal evaluation.
2. **Missing Checkpoint and Raw Prediction Persistence:** In Phase 4.4, checkpoints and sample-level prediction vectors were discarded from disk.
3. **Domain Imbalance Masking:** The test set was 86.31% synthetic turntable images (`fruits360`), masking real-world grading accuracy (<11%).
4. **Extreme Loss Weighting Instability:** Raw inverse weighting ($cw_k \propto 1/N_k$) caused $111\times$ weight disparities, triggering catastrophic false-alarm explosions on minority grades for sensitive initializations (Seeds 19 and 31).
5. **Collinear and Negative Contexts:** OOD and Uncertainty features exhibited exact collinearity ($r = -1.0$) and negative marginal contributions.

This document formally pre-registers the **Phase 5.0 Confirmatory Experimental Protocol**.

---

## 2. Experimental Configurations and Hypotheses

The benchmark evaluates a 4-way factorial design across identical data partitions, seeds, and optimization budgets:

| Config ID | Model Designation | Input Features | Rationale |
|:---:|---|---|---|
| **A** | Image-Only Baseline | ConvNeXt visual embedding $\mathbf{x} \in \mathbb{R}^{768}$ | Standard computer vision baseline without metadata. |
| **B** | Image + Species (Recommended) | $[\mathbf{x}; \mathbf{e}_s] \in \mathbb{R}^{784}$ | Primary hypothesis: species embedding constrains threshold drift. |
| **C** | Image + OOD | $[\mathbf{x}; \mathbf{c}_{\text{ood}}] \in \mathbb{R}^{770}$ | Controls for open-set feature contribution alone. |
| **D** | Image + Species + OOD | $[\mathbf{x}; \mathbf{e}_s; \mathbf{c}_{\text{ood}}] \in \mathbb{R}^{786}$ | Full multimodal system (CASM v2). |

### Pre-Registered Primary and Secondary Endpoints:
- **Primary Confirmatory Endpoint:**
  $$\mathbf{H_1}: \mu_{\text{QWK}}(\text{Config B}) > \mu_{\text{QWK}}(\text{Config A})$$
  Evaluated using a **paired two-sided Student's $t$-test** at prespecified $\alpha = 0.05$ across $N=30$ paired seeds.
- **Sensitivity Primary Test:** Two-sided paired Wilcoxon signed-rank test.
- **Secondary Endpoints:**
  1. Mean Absolute Error (MAE)
  2. Off-By-One (OBO) Accuracy
  3. Accuracy
  4. 5-Class Macro-F1
- **Multiplicity Adjustment:** Holm-Bonferroni step-down procedure controlling Family-Wise Error Rate (FWER) at $\alpha = 0.05$ across the 4 secondary endpoints.

---

## 3. Dataset Partitions and Split Protocols

To ensure complete scientific integrity, **historical results and corrected-split results must never be pooled**. They must be evaluated and reported in separate, distinct tables:

### 1. Partition Benchmark 1: Historical Split (`split_v1`)
- Preserves the exact split in `master_dataset.csv` (Train: 39,909, Val: 3,413, Test: 12,714).
- Grade 2 has 0 test samples. Evaluated strictly as a 4-active-class benchmark $\{0, 1, 3, 4\}$ to reproduce historical baselines.

### 2. Partition Benchmark 2: Corrected Split (`split_v2`)
- Preserves specimen and session-level grouping (0 cross-split path leakage, 0 cross-split content duplicate hashes).
- Reallocates session `20200730` ($n=119$ Banana Grade 2 images) from `train` to `test`.
- Guaranteed Grade Coverage:
  * Train: Grade 0 (36,414), Grade 1 (846), Grade 2 (208), Grade 3 (1,606), Grade 4 (716) -> Total: 39,790
  * Val: Grade 0 (1,037), Grade 1 (494), Grade 2 (196), Grade 3 (1,268), Grade 4 (418) -> Total: 3,413
  * Test: Grade 0 (11,605), Grade 1 (88), **Grade 2 (119)**, Grade 3 (868), Grade 4 (153) -> Total: 12,833
- Enables the first genuine 5-grade ordinal evaluation.

### 3. Partition Benchmark 3: Real-World-Only Evaluation Split (`split_real`)
- Filters out all $43,962$ synthetic `fruits360` images.
- Evaluates models strictly on real-world market images ($N = 12,074$) to measure true agricultural utility without turntable background bias.

---

## 4. Model Architecture and Loss Function Specification

### 1. Architectural Standardization:
- **Visual Backbone:** Frozen ConvNeXt-Tiny pre-trained embeddings ($768$-D).
- **Visual Projection:**
  $$\text{Linear}(768, 512) \to \text{LayerNorm}(512) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(512, 256) \to \text{LN}(256) \to \text{GELU}$$
- **Species Embedding:** `RobustSpeciesEmbedding(embedding_dim=16)` with 5 known species and index 5 reserved for `UNKNOWN_SPECIES`.
- **Fusion MLP:**
  $$\text{Linear}(256 + D_{\text{ctx}}, 256) \to \text{LN}(256) \to \text{GELU} \to \text{Dropout}(0.2) \to \text{Linear}(256, 128) \to \text{LN}(128) \to \text{GELU} \to \text{Dropout}(0.1)$$
- **Output Head:** `MonotonicCORALHead(128, num_classes=5)` from `casm_v2_corrected_modules.py`. Structurally guarantees threshold monotonicity and non-negative class probabilities.

### 2. Class Weighting Reform:
To eliminate the instability caused by raw inverse class weighting ($24.4\times$ weight ratio), the loss shall adopt **Effective Number of Samples Weighting** (Cui et al., CVPR 2019):
$$cw_k = \frac{1 - \beta}{1 - \beta^{N_k}}, \quad \text{with } \beta = 0.999$$
This smooths the extreme weight ratio from $111.4:1$ down to a bounded $\approx 8.5:1$, preventing false-alarm explosions on minority classes while still providing adequate gradient signal.

---

## 5. Training, Optimization, and Artifact Persistence Rules

### 1. Optimization Parameters:
- **Optimizer:** `torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)`
- **Learning Rate Schedule:** `CosineAnnealingLR(T_max=20, eta_min=1e-5)`
- **Batch Size:** 1,024
- **Epoch Budget:** 20 epochs
- **Model Selection Criterion:** Model checkpoint with the lowest validation MAE (`val_mae`) is saved. Validation data must NEVER be used to choose test-time evaluation thresholds.

### 2. Pre-Registered Seed Schedule ($N=30$):
```python
SEEDS = [
    42, 123, 777, 999, 2025,
    1, 2, 3, 5, 7,
    11, 13, 17, 19, 23,
    31, 37, 41, 43, 47,
    53, 59, 61, 67, 71,
    73, 79, 83, 89, 97
]
```
All seeds must be executed deterministically. No seed may be dropped or excluded from the primary confirmatory analysis.

### 3. Mandatory Artifact Persistence:
For every completed run $(m, s) \in \{\text{A, B, C, D}\} \times \text{SEEDS}$:
1. **Checkpoint:** Save model state dict to `outputs/phase5/checkpoints/model_{cfg}_seed{s}_best.pth`.
2. **Raw Predictions:** Save full integer prediction vector $\mathbf{\hat{y}} \in \mathbb{Z}^{N_{\text{test}}}$ and class probability tensor $\mathbf{P} \in \mathbb{R}^{N_{\text{test}} \times 5}$ to `outputs/phase5/predictions/preds_{cfg}_seed{s}.pt`.
3. **Per-Seed Metric Record:** Append row to `outputs/phase5/phase5_benchmark_raw.csv`.

---

## 6. Stage 6 Sign-Off Verdict

- **Delivered:** Fully specified, publication-defensible experimental protocol.
- **Addressed:** Grade 2 test representation; monotonic probability guarantee; robust species fallback; effective number of samples weighting; full persistence of checkpoints and raw prediction vectors.
- **Safety to Proceed:** It is **SAFE** to proceed to Phase 7 (Audit OOD and Calibration Evidence).
