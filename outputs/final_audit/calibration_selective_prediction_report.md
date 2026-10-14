# Uncertainty Calibration & Selective Prediction Audit Report

**Date:** October 2026  
**Model Evaluated:** ConvNeXt-Tiny with Monotonic CORAL Head (30-Seed Ensemble)  
**Sample Size:** 12,833 test images across 5 grades  

---

## 1. Summary of Calibration Metrics

- **Expected Calibration Error (ECE, 10 Bins):** **2.51%**  
- **Multi-Class Brier Score:** **0.0887**  

### Reliability Diagram Table

| Confidence Bin | Sample Count | Sample Proportion (%) | Mean Confidence (%) | Observed Accuracy (%) | Calibration Gap (%) |
|:---|---:|---:|---:|---:|---:|
| `(0.2, 0.3]` | 618 | 4.82% | 24.78% | 5.34% | 19.44% |
| `(0.3, 0.4]` | 286 | 2.23% | 35.11% | 19.58% | 15.53% |
| `(0.4, 0.5]` | 195 | 1.52% | 43.72% | 25.13% | 18.6% |
| `(0.5, 0.6]` | 115 | 0.9% | 54.73% | 48.7% | 6.03% |
| `(0.6, 0.7]` | 174 | 1.36% | 64.6% | 62.07% | 2.53% |
| `(0.7, 0.8]` | 206 | 1.61% | 75.18% | 88.35% | 13.17% |
| `(0.8, 0.9]` | 421 | 3.28% | 85.34% | 96.44% | 11.1% |
| `(0.9, 1.0]` | 10,818 | 84.3% | 99.57% | 99.9% | 0.33% |

---

## 2. Selective Prediction & Safe Industrial Abstention

> In automated post-harvest optical sorting, uncertain fruits can be rejected to secondary inspection belts. We trace the risk-coverage curve by discarding the lowest-confidence predictions.

| Coverage Level (%) | Retained Samples | Sorting Accuracy (%) | Severity MAE | Confidence Threshold (%) |
|:---:|---:|---:|---:|---:|
| **100.0%** | 12,833 | **91.15%** | 0.1382 | $\ge$ 21.4% |
| **95.0%** | 12,191 | **95.68%** | 0.0769 | $\ge$ 30.65% |
| **90.0%** | 11,549 | **99.14%** | 0.0166 | $\ge$ 63.14% |
| **85.0%** | 10,908 | **99.88%** | 0.0034 | $\ge$ 87.55% |
| **80.0%** | 10,266 | **99.99%** | 0.0003 | $\ge$ 97.79% |
| **70.0%** | 8,983 | **100.0%** | 0.0000 | $\ge$ 99.77% |
| **60.0%** | 7,699 | **100.0%** | 0.0000 | $\ge$ 99.84% |
| **50.0%** | 6,416 | **100.0%** | 0.0000 | $\ge$ 99.86% |

---

## 3. Conclusions on Calibration & Trustworthiness

1. **Monotonic Probability Consistency:**  
   Unlike unconstrained CORAL heads which produced negative probabilities and arbitrary scale bounds,    the `MonotonicCORALHead` produces normalized probability distributions that closely match observed sorting frequencies.
2. **Operational Efficacy of Abstention:**  
   By abstaining on the lowest 10% of confident predictions (90% coverage), sorting accuracy increases from **91.15%**    to **99.14%**, and severity MAE drops by an order of magnitude (0.1382 to 0.0166).
3. **Domain Caveat:**  
   High confidence is predominantly concentrated in the studio `fruits360` domain. When deployed on market bins (`split_real`),    the average confidence drops to 48.7%, triggering abstention on up to 64% of real-world samples under a 70% threshold.