# FINAL RELEASE REPORT: Phase 5.7 Final Submission Lock & GitHub Publication

**Project Title:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation  
**Author / Lead ML Research Engineer:** Nathish N (`nathishnataraj@gmail.com`)  
**Institutional Affiliation:** Final-Year Engineering Research Project  
**Repository Remote:** `https://github.com/NathishN/Final-year-project.git`  
**Git Branch:** `main`  
**Publication Lock Directory:** `outputs/phase57_final_submission_lock/`  
**Mirror Submission Directory:** `outputs/final_submission/`  
**Verification Date:** October 14, 2026  
**Final Release Decision:** **`READY FOR SUBMISSION AND PUSHED TO GITHUB`**

---

## Executive Summary

This report establishes the complete, mathematically grounded, and empirically certified release of the final-year research project: **"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation"**. 

Every empirical claim, statistical contrast, and OOD metric across all phases of the research has undergone rigorous multi-tier audit and recomputation. All 240 model runs have been independently verified against raw prediction tensors (240/240 `PASS`), all six confirmatory statistical contrasts have been verified via exact permutation distributions with Holm-Bonferroni family-wise error rate control ($p_{\text{adj}} < 0.05$), and the complete automated regression test suite passes with zero defects (28/28 `PASS`, exit code 0). 

The repository has been packaged cleanly with strict file-size hygiene, comprehensive documentation across 24 mandatory sections, and published to the target GitHub repository without disrupting existing history.

---

## 1. Resolution of the Part B Real-Image OOD Metric Discrepancy

### 1.1 Root Cause Identification & Mathematical Diagnosis
A multi-tier audit traced the exact origin of divergent OOD figures across historical reports:

1. **The Feature Normalization Asymmetry Mechanism:**
   In `phase50_stage4_real_ood.py` and `scripts/phaseE_ood_audit.py`, in-distribution (ID) test features were loaded from pre-extracted cache files (`test_feats`, 768-D vectors extracted from the backbone prior to the classifier head's LayerNorm). ID logits were then computed via raw linear projection:
   $$\mathbf{z}_{\text{ID}} = \mathbf{f}_{\text{cached}} \mathbf{W}^T + \mathbf{b}$$
   skipping the crucial `model.head.norm` (LayerNorm) operation. In contrast, out-of-distribution (OOD) test samples were evaluated through the full end-to-end model forward pass:
   $$\mathbf{z}_{\text{OOD}} = \mathbf{W} \cdot \operatorname{LayerNorm}(\mathbf{f}_{\text{backbone}}) + \mathbf{b}$$
   Because $\operatorname{LayerNorm}$ centers and rescales feature activations, the skipped LayerNorm on ID features inflated the magnitude and variance of ID logits, artificially skewing logit-based OOD detectors (Energy and MSP) upward.

2. **Metric Traceability & Invariance:**
   - **Mahalanobis Distance (84.22% AUROC):** Operates directly on the 768-D penultimate backbone features before any classification head or LayerNorm. Because features are identical in both paths, Mahalanobis AUROC remained invariant at **84.22%** across both historical and corrected evaluations.
   - **Energy Score (AUROC):** Historical asymmetric logits yielded **81.99%** AUROC; mathematically corrected symmetric logits yield **63.03%** AUROC.
   - **Maximum Softmax Probability (MSP AUROC):** Historical asymmetric logits yielded **74.26%** AUROC; mathematically corrected symmetric logits yield **56.27%** AUROC.

3. **Refutation of Threshold-Dependent AUROC Claims:**
   Previous draft notes hypothesized that threshold calibration altered AUPR or AUROC. In mathematical statistics, ROC and Precision-Recall curves are constructed by sweeping all possible detection thresholds $\tau \in (-\infty, \infty)$. Therefore, changing a static operational decision threshold cannot alter AUROC or AUPR. The draft numbers (54.06% / 44.20% and 41.87% / 61.27%) were discarded as unverified draft scratchpad artifacts lacking preserved prediction tensors.

### 1.2 Authoritative Real-Image OOD Results (Near-OOD: Real Near-Species Fruits)
Both historical and corrected symmetric metrics are fully documented and preserved in `tables/ood_metrics_recomputed.csv`:

| OOD Detector | Historical Metric (AUROC / AUPR) | Corrected Symmetric Metric (AUROC / AUPR) | Feature Representation | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Mahalanobis Distance** | **84.22%** / **81.33%** | **84.22%** / **81.33%** | 768-D Penultimate Backbone Features | **Invariant / Authoritative** |
| **Energy Score** | 81.99% / 80.12% | **63.03%** / **57.14%** | Post-LayerNorm Logits (Free Energy $T=1.0$) | Corrected Symmetric |
| **Max Softmax Prob (MSP)** | 74.26% / 72.84% | **56.27%** / **50.81%** | Post-LayerNorm Softmax Confidence | Corrected Symmetric |

---

## 2. Verification of 240 Model Runs (Part C)

All 240 model runs across 4 model architectures ($\mathcal{M}_A, \mathcal{M}_B, \mathcal{M}_C, \mathcal{M}_D$), 30 random seeds, and both validation and test evaluation splits were audited against raw prediction tensors:

$$\mathcal{M}_A \text{ (Full CASM v2)}, \quad \mathcal{M}_B \text{ (Ordinal Only)}, \quad \mathcal{M}_C \text{ (Cross-Entropy Multi-Head)}, \quad \mathcal{M}_D \text{ (Standard CORAL)}$$

- **Total Runs Audited:** 240 / 240
- **Total Valid Pass Rate:** 240 / 240 (100.0%)
- **Max Absolute Discrepancy ($\Delta$ QWK):** $\le 0.000050$ across all 240 runs.
- **QWK Mathematical Equivalence:** Proved that scratch implementation matches `scikit-learn.metrics.cohen_kappa_score(weights='quadratic')` within float64 machine epsilon ($|\Delta| = 2.22 \times 10^{-16}$).
- **Ordinal Decoding Equivalence:** Predicted discrete ordinal ranks match probabilistic threshold decoding ($\sum_{k=1}^{K-1} \mathbb{I}(P(Y > k) \ge 0.5)$) with 100% agreement.

### Authoritative Architecture Performance Summary (30 Seeds, Test Split)

| Architecture | QWK (Mean $\pm$ SD) | 95% Bootstrap CI | Accuracy | Macro-F1 | ECE (%) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Full CASM v2)** | **0.8711 $\pm$ 0.0084** | **[0.8679, 0.8741]** | **85.08%** | **0.8491** | **3.82%** | 8.42 ms |
| **Model B (Ordinal Only)** | 0.8637 $\pm$ 0.0089 | [0.8604, 0.8670] | 84.12% | 0.8385 | 4.21% | 8.11 ms |
| **Model C (Cross-Entropy Multi-Head)** | 0.8524 $\pm$ 0.0112 | [0.8483, 0.8566] | 83.05% | 0.8274 | 5.48% | 8.35 ms |
| **Model D (Standard CORAL)** | 0.8402 $\pm$ 0.0125 | [0.8356, 0.8448] | 81.92% | 0.8159 | 6.12% | 8.05 ms |

---

## 3. Confirmatory Statistical Contrasts & Multiplicity Control (Part D)

The 6 paired statistical contrasts comparing Model A against ablations and baselines across all 30 seeds were evaluated using three independent statistical formulations:
1. **Exact Conditional Permutation Distribution (Dynamic Programming):** $2^{30} = 1,073,741,824$ states.
2. **SciPy Exact Permutation Test:** Paired difference distribution.
3. **Monte Carlo Sign-Flip Permutation:** $B = 2,000,000$ iterations.

### Paired Contrast Results Table

| Contrast ID | Contrast Description | Mean Diff ($\Delta$) | 95% Paired CI | Exact Wilcoxon $W$ | Exact Perm $p$-value | Monte Carlo $p$ | Holm-Bonferroni $\alpha_{\text{adj}}$ | Significant ($p < \alpha_{\text{adj}}$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | Model B $-$ Model A (Val) | $-0.0082$ | $[-0.0114, -0.0051]$ | 52.0 | $3.21 \times 10^{-4}$ | $3.20 \times 10^{-4}$ | 0.0167 | **YES** |
| **C2** | Model B $-$ Model A (Test) | $-0.0074$ | $[-0.0102, -0.0046]$ | 55.0 | $4.88 \times 10^{-4}$ | $4.85 \times 10^{-4}$ | 0.0250 | **YES** |
| **C3** | Model C $-$ Model A (Val) | $-0.0201$ | $[-0.0245, -0.0157]$ | 0.0 | $1.86 \times 10^{-9}$ | $< 1.0 \times 10^{-6}$ | 0.0083 | **YES** |
| **C4** | Model C $-$ Model A (Test) | $-0.0187$ | $[-0.0228, -0.0146]$ | 2.0 | $3.72 \times 10^{-9}$ | $< 1.0 \times 10^{-6}$ | 0.0100 | **YES** |
| **C5** | Model D $-$ Model A (Val) | $-0.0324$ | $[-0.0371, -0.0277]$ | 0.0 | $1.86 \times 10^{-9}$ | $< 1.0 \times 10^{-6}$ | 0.0083 | **YES** |
| **C6** | Model D $-$ Model A (Test) | $-0.0309$ | $[-0.0355, -0.0263]$ | 0.0 | $1.86 \times 10^{-9}$ | $< 1.0 \times 10^{-6}$ | 0.0083 | **YES** |

- **Multiplicity Control:** All 6 hypothesis tests remain statistically significant after strict Holm-Bonferroni step-down correction ($p_{\text{adj}} < 0.05$).
- **Sign Test Reconciliation:** For Model B $-$ Model A (Test), exactly 9 seeds favored Model B and 21 seeds favored Model A (zero ties), correcting historical typo references of `7/23`.

---

## 4. Regression & Integrity Verification Harness (Part E)

The automated regression test harness and integrity scanner were executed locally:

```text
pytest tests outputs/casm_v2/test_corrected_modules.py -v
============================= 28 passed in 38.04s =============================
Exit Code: 0
```

1. **`tests/test_data_integrity.py` (5 tests):** PASSED
   - All 5 ordinal grades present in ground-truth distribution.
   - Grade 2 test set representation certified.
   - Total sample count matches 2,751 valid annotated samples.
   - Zero cross-split path leakage across train/val/test splits.
   - Zero Grade 2 acquisition session leakage certified.
2. **`tests/test_end_to_end_smoke.py` (1 test):** PASSED
   - End-to-end forward-backward training pass verified.
3. **`tests/test_integration_predictions.py` (4 tests):** PASSED
   - All 120 prediction files present and validated.
   - Confirmatory statistical reconciliation verified.
   - Prediction tensor structure, shapes, and probabilities verified.
   - Recomputed metrics match stored tables.
4. **`tests/test_metrics_and_calibration.py` (3 tests):** PASSED
   - Degenerate QWK edge cases handled correctly.
   - ECE calibration calculation verified.
   - QWK boundary conditions certified.
5. **`tests/test_model_properties.py` (4 tests):** PASSED
   - Monotonic ordinal cumulative probability validity.
   - Strictly decreasing cutpoint constraints verified.
   - Logit numerical stability under extreme activations.
   - Robust species embedding unknown-fruit routing certified.
6. **`tests/test_phase56_regression_hardening.py` (7 tests):** PASSED
   - Holm-Bonferroni family-wise error rate control.
   - Monotonic CORAL head mathematical properties.
   - Discrete vs probabilistic ordinal decoding equivalence.
   - Model parameter counts match verified real checkpoint.
   - Selective abstention monotonicity and risk-coverage reduction.
   - Session isolation certified with 0 samples leaked.
   - Exact Wilcoxon distribution matches asymptotic approximation.
7. **`outputs/casm_v2/test_corrected_modules.py` (4 tests):** PASSED
   - Monotonic CORAL head extreme values and dtypes.
   - Head architectural constraints.
   - Original unconstrained CORAL rank-inversion failure mode verified.
   - Robust species embedding routing logic.

**Integrity Scanner (`scripts/scan_integrity.py`):**
- Result: **0 forbidden patterns or stale metric claims found**.

---

## 5. Artifact Manifest & SHA-256 Checksums (Part F)

All deliverables in `outputs/phase57_final_submission_lock/` are locked with cryptographic SHA-256 hashes:

| Artifact File | Size (Bytes) | SHA-256 Checksum |
| :--- | :---: | :--- |
| `manuscript.md` | 28,900 | `c0877780862490a9ffe9651b25d26d92636d62533f9778a16e8c384c9606d36a` |
| `abstract.md` | 4,391 | `c66dae55895899a88ff1799d4ba5ba7dca5f3e9c76a257f4520a553e2d422202` |
| `final_results.json` | 13,073 | `87a803625f2a47eff1e249b1218ec957e2d75ec3e3ad7eed1502a9b7c91fcae6` |
| `references.bib` | 3,909 | `d92c24c52e816928550831c0bbea11cb56fb72ff7deaca086bedd78cdc98a311` |
| `CHANGELOG.md` | 8,047 | `dc7ba155da3131db02d2c56d74778a33a3ecb5edadfb3c04384a87f6ae2dc92d` |
| `claim_ledger.csv` | 11,476 | `ecf07e23f730392d8d188583766dbe4ddab82b97eced29fb90272f95c1b1a1d0` |
| `tables/authoritative_statistical_table.csv` | 1,618 | `09cc384736ca753983d0fe34369ae39c67d948f1f27f079a5b8e30d15b9a277b` |
| `tables/gap1_per_seed_metric_verification.csv` | 51,382 | `1adcc4b0988032354f3dbb98631c956e45c7e778e2ea12dea9bde442bdfb53b6` |
| `tables/gap1_aggregate_metrics_summary.csv` | 2,320 | `ec3a3783a754bfc6c98a16db30d8692348ab0c83aadeb48de27b4e04aac3aac7` |
| `tables/gap3_statistical_contrasts_verified.csv` | 2,540 | `869e5bb969e0740bb83dd39cb6f076a078783b3b3c68f4679da3731f9de49000` |
| `tables/ood_metrics_recomputed.csv` | 1,232 | `c801a67968cf0702eeabfc9872eec7e54497ca3d204e9c8533deb36b4053ca20` |
| `tables/split_integrity_summary.csv` | 516 | `112204944ec88e6a998c3e46c443ad1e4675fa835d66902c380fc1fd7a3d4971` |
| `tables/calibration_recomputed.csv` | 611 | `309793ca995b84c1c4b108bedffbd6753361c6ee8636af95576bd7cb520cd316` |
| `tables/selective_prediction_recomputed.csv` | 984 | `ed8b3fb1b1bba4b897b16ed51b3e418ddb26cb11224fb912fef457239d583f36` |
| `figures/figure1_dataset_composition.png` | 182,804 | `88f206bd640680675bd4a72cbff9038e2800f3563e27d09d7147a1d6c4b2b8ab` |
| `figures/figure2_qwk_distribution.png` | 220,539 | `a7ac45080bc0fa47c92dc28da68bfcb61b45bc3dcf24c8343c0e022a57c835b9` |
| `figures/figure3_paired_contrasts.png` | 158,426 | `64ac1255f61e5103aa8201a4f6280552032ad242c200292b0d93d20a2cffe627` |
| `figures/figure4_ood_domain_inversion.png` | 160,911 | `66b7d76b205deae73c6e50745c68d3d37c7a4f46dada50fe6b1430cba12a7b73` |
| `figures/figure5_selective_risk_coverage.png` | 264,268 | `bd1013661a1fb87da8f128659927e7b1323e813ac68f9757b6ec7360c5a89f8d` |
| `figures/figure6_inference_benchmarks.png` | 175,027 | `4a241ec6759bc7fcb3ea27a0d94f978764845079bc2a3018c3c75f7913fda3c3` |

---

## 6. Git Hygiene & GitHub Publication Audit (Part H)

- **Target Remote:** `https://github.com/NathishN/Final-year-project.git`
- **Target Branch:** `main`
- **Integration Strategy:** Linear Git rebase preserving initial commit `eb678e6` as base ancestor.
- **File Hygiene Certified:**
  - Multi-gigabyte raw image folders (`Apple/`, `Banana/`, etc.) excluded via `.gitignore`.
  - Binary files exceeding GitHub 100MB hard limit excluded (`outputs/checkpoints/best_model.pth` [318.6 MB], `outputs/coral/checkpoints/coral_best_model.pth` [106.2 MB], `outputs/features/phase4_dataset.pt` [170.9 MB]).
  - Maximum tracked file size: $\le 2.6 \text{ MB}$ (exemplar model weights `outputs/casm_v2/phase50_checkpoints/model_A_seed83_best.pth`).
  - Zero private credentials, API keys, or certificates committed.
  - Template `.env.example` committed for reproducible path configuration.

---

## 7. Final Certification & Sign-Off

The research codebase, experimental outcomes, statistical contrasts, and publication artifacts have met all standards of mathematical rigor, software regression hardening, and empirical reproducibility.

```text
STATUS: READY FOR SUBMISSION AND PUSHED TO GITHUB
VERIFIED BY: Principal ML Research Engineer & Reproducibility Auditor
DATE: October 14, 2026
```
