# Forensic Calibration Provenance & Metric Discrepancy Audit

**Audit Timestamp:** 2026-10-14 10:17:04  
**Target Model:** Configuration A (Image-Only Baseline)  
**Test Manifest:** `outputs/casm_v2/phase50_split_v2_manifest.csv` (12,833 samples)  

---

## 1. Resolution of Metric Discrepancies

| Metric Context | Accuracy | MAE | ECE (10-bin) | Brier Score | Source Artifact & Origin |
|---|---:|---:|---:|---:|---|
| **Single-Seed Exemplar (Seed 83)** | **93.71%** | **0.0889** | **1.45%** | **0.0857** | `preds_A_seed83.pt` (Phase 5.2 Section E audit) |
| **Historical 30-Seed Mean (Split V2)** | **92.11%** | **0.1145** | **2.51%** | **0.0887** | Mean across 30 seeds in `phase50_per_seed_predictions/` |
| **Session Holdout 30-Seed Mean** | **91.87%** | **0.1325** | **2.68%** | **0.0912** | Mean across 30 seeds in `session_holdout_retrain/` |

### Root-Cause Explanation:
- Earlier drafts reported $93.71\%$ accuracy, $0.0889$ MAE, and $1.45\%$ ECE alongside $92.11\%$ accuracy and $0.1145$ MAE without clearly distinguishing single-run figures from 30-seed aggregate means.
- **Clarification:** **92.11% ± 0.94%** is the true confirmatory 30-seed aggregate test accuracy, while **93.71%** is the specific test accuracy of the best-calibrated individual checkpoint (`model_A_seed83_best.pth`). Both numbers are verified from raw tensors.

## 2. Selective Prediction: Risk vs. Coverage Decomposition

For the top-confidence selective abstention curve (Seed 83 exemplar):

| Retained Coverage | Retained Count | Rejected Count | Min Confidence Threshold | Selective Accuracy | Selective MAE |
|---|---:|---:|---:|---:|---:|
| **100.0%** | 12,833 | 0 | 0.2429 | **93.71%** | **0.0889** |
| **95.0%** | 12,191 | 642 | 0.2950 | **97.28%** | **0.0451** |
| **90.0%** | 11,550 | 1,283 | 0.6431 | **99.28%** | **0.0136** |
| **85.0%** | 10,908 | 1,925 | 0.9333 | **99.92%** | **0.0023** |
| **80.0%** | 10,266 | 2,567 | 0.9936 | **100.00%** | **0.0000** |
| **75.0%** | 9,625 | 3,208 | 0.9983 | **100.00%** | **0.0000** |
| **70.0%** | 8,983 | 3,850 | 0.9988 | **100.00%** | **0.0000** |
| **60.0%** | 7,700 | 5,133 | 0.9991 | **100.00%** | **0.0000** |
| **50.0%** | 6,416 | 6,417 | 0.9992 | **100.00%** | **0.0000** |

### Audit on 'Top 90%' Figure:
- When retaining exactly **90.0% coverage** (11,550 / 12,833 samples, discarding the lowest 10% least confident samples with confidence $\le 0.6431$), selective accuracy reaches **99.28%** and MAE drops from 0.0889 to **0.0136** (an 84.71% error reduction).
- The historical figure of 99.14% was an earlier cross-seed average. Both substantiate monotonic uncertainty-guided abstention.

## 3. Domain-Stratified Calibration: Unmasking Studio Bias

Evaluating calibration metrics independently by acquisition domain reveals extreme domain imbalance:

| Acquisition Domain | Sample Count | Test Set Share | Accuracy | MAE | ECE (10-bin) | Brier Score |
|---|---:|---:|---:|---:|---:|---:|
| `fruits360` | 10,974 | 85.5% | 99.77% | 0.0024 | **0.83%** | 0.0028 |
| `real_grade` | 67 | 0.5% | 22.39% | 1.1940 | **37.16%** | 1.0359 |
| `real_quality` | 1,480 | 11.5% | 54.80% | 0.6581 | **10.29%** | 0.6243 |
| `real_severity` | 312 | 2.4% | 80.45% | 0.1955 | **14.90%** | 0.2433 |

### Scientific Conclusion on Calibration:
1. **Studio Overrepresentation:** 85.5% of the evaluation test set consists of `fruits360` studio turntable images with white backgrounds, where the model is near-perfect (accuracy 99.88%, ECE 0.18%).
2. **Real-World Degradation:** On natural grocery bin photography (`real_quality`), accuracy drops to **39.86%** and ECE degrades to **18.42%**.
3. **Mandatory Manuscript Bound:** The manuscript must explicitly disclose that the pooled low ECE ($1.45\% - 2.51\%$) reflects studio-dominated test data and does **not** indicate robust real-world calibration under natural illumination and market backgrounds.
