# Phase 5.2 — Session-Held-Out Evaluation & Retraining Report

## 1. Overview and Motivation

In the historical `split_v2` protocol:
- **119 Banana Grade 2 images** from session `20200730` were assigned to `test` to enable minority Grade 2 evaluation.
- **147 non-Grade-2 images** (Apple Grade 0: 15, Banana Grade 0: 85, Banana Grade 3: 47) from session `20200730` remained in `train`.

While zero image paths and zero hashes overlapped between splits, complete session-level holdout was not achieved because session `20200730` spanned both train and test partitions.

To establish rigorous group-held-out scientific validity, Phase 5.2 executed:
1. **Manifest Reassignment:** Reassigned all 266 samples of session `20200730` to `test` (`session_holdout_manifest.csv`), establishing **100% group isolation across all 80 acquisition sessions** (0 cross-split sessions; Train: 39,643, Val: 3,413, Test: 12,980).
2. **Full 120-Run Retraining Campaign:** Executed 120 independent runs (4 configurations $\times$ 30 prespecified seeds) from scratch under the strictly session-isolated manifest, using Effective Number class weighting calculated strictly on the training partition and validation QWK checkpoint selection. All historical artifacts were preserved untouched in `outputs/casm_v2/`, and new artifacts were stored in `outputs/phase52_final_validation/session_holdout_retrain/`.

---

## 2. Results of 120-Run Group Session-Held-Out Retraining

| Configuration | Architecture | Test QWK (Mean ± SD) | Test MAE | Test Accuracy (%) | Macro-F1 (%) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **Config A** | Image-Only Baseline | **0.8038 ± 0.0335** | **0.1325** | **91.87%** | **48.21%** |
| **Config B** | Image + Species | 0.7776 ± 0.0431 | 0.1433 | 91.58% | 45.98% |
| **Config C** | Image + OOD | 0.7513 ± 0.0352 | 0.1635 | 89.57% | 41.92% |
| **Config D** | Image + Species + OOD | 0.7164 ± 0.0494 | 0.1725 | 89.83% | 42.49% |

---

## 3. Paired Statistical Contrasts (30 Seeds)

| Contrast | Mean Difference | 95% Confidence Interval | Cohen's $d_z$ | Paired $t(29)$ | Raw $p$ | Holm-Bonferroni $p_{adj}$ | Wilcoxon $W$ | Wilcoxon $p$ |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|
| **B vs A (Species Effect)** | **-0.0262** | [-0.0456, -0.0067] | -0.5019 | -2.7490 | 0.01018 | **0.01018** | 125.0 | **0.02703** |
| **C vs A (OOD Effect)** | **-0.0525** | [-0.0707, -0.0342] | -1.0720 | -5.8715 | $2.27 \times 10^{-6}$ | **$4.53 \times 10^{-6}$** | 29.0 | **$3.24 \times 10^{-6}$** |
| **D vs A (Species + OOD)** | **-0.0874** | [-0.1081, -0.0667] | -1.5744 | -8.6231 | $1.70 \times 10^{-9}$ | **$5.09 \times 10^{-9}$** | 9.0 | **$6.15 \times 10^{-8}$** |

---

## 4. Key Scientific Findings

1. **Negative Result Replicates Under Strict Group Holdout:** Even when models are retrained completely from scratch with zero session leakage, species conditioning and OOD context conditioning statistically degrade ordinal grading performance relative to the Image-Only baseline ($p < 0.05$ for B vs A; $p < 10^{-5}$ for C vs A and D vs A).
2. **No Data Leakage Confound:** The negative finding is not an artifact of session 20200730 spanning train and test in `split_v2`. Both historical and group-isolated protocols independently confirm the identical conclusion.
