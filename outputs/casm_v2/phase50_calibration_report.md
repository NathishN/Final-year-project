# Phase 5.0 -- Stage 7: Uncertainty Calibration & Selective Prediction Report

**Date:** October 2026  
**Model Evaluated:** CASM v2 Monotonic CORAL Model (Ensemble across 30 seeds)  

---

## 1. Probability Calibration Metrics

- **Expected Calibration Error (ECE, 10 Bins):** 2.51%  
- **Multi-Class Brier Score:** 0.0887  

### Reliability Diagram Breakdown

| Confidence Bin | Sample Count | Pct (%) | Avg Confidence (%) | Observed Accuracy (%) | Calibration Gap (%) |
|:---|---:|---:|---:|---:|---:|
| (0.2, 0.3] | 618 | 4.82% | 24.78% | 5.34% | 19.44% |
| (0.3, 0.4] | 286 | 2.23% | 35.11% | 19.58% | 15.53% |
| (0.4, 0.5] | 195 | 1.52% | 43.72% | 25.13% | 18.6% |
| (0.5, 0.6] | 115 | 0.9% | 54.73% | 48.7% | 6.03% |
| (0.6, 0.7] | 174 | 1.36% | 64.6% | 62.07% | 2.53% |
| (0.7, 0.8] | 206 | 1.61% | 75.18% | 88.35% | 13.17% |
| (0.8, 0.9] | 421 | 3.28% | 85.34% | 96.44% | 11.1% |
| (0.9, 1.0] | 10,818 | 84.3% | 99.57% | 99.9% | 0.33% |

---

## 2. Selective Prediction & Safe Abstention (Risk-Coverage Profile)

> When deployed in industrial sorting conveyors, the model can abstain on low-confidence predictions, routing uncertain fruit to manual inspection stations.

| Coverage (%) | Retained Samples | Sorting Accuracy (%) | Severity MAE | Confidence Threshold (%) |
|:---:|---:|---:|---:|---:|
| 100.0% | 12,833 | **91.15%** | 0.1382 | ≥ 21.4% |
| 95.0% | 12,191 | **95.68%** | 0.0769 | ≥ 30.65% |
| 90.0% | 11,549 | **99.14%** | 0.0166 | ≥ 63.14% |
| 85.0% | 10,908 | **99.88%** | 0.0034 | ≥ 87.55% |
| 80.0% | 10,266 | **99.99%** | 0.0003 | ≥ 97.79% |
| 70.0% | 8,983 | **100.0%** | 0.0000 | ≥ 99.77% |
| 60.0% | 7,699 | **100.0%** | 0.0000 | ≥ 99.84% |
| 50.0% | 6,416 | **100.0%** | 0.0000 | ≥ 99.86% |

---

## 3. Conclusions on Uncertainty

1. **Monotonic Probability Consistency:**  
   The MonotonicCORALHead produces mathematically consistent class probabilities that sum to 1.0 without negative probabilities.
2. **Well-Behaved Calibration:**  
   The ECE of 2.51% demonstrates that model confidence closely aligns with empirical sorting accuracy.
3. **Practical Utility of Selective Abstention:**  
   By abstaining on the lowest 20% of confident samples (80% coverage), sorting accuracy improves from 91.15% to 99.99%, and MAE drops significantly.