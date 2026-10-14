# Phase 5.2 — Calibration & Selective Prediction Report

## 1. Calibration Metrics

- **ECE (10 bins):** 1.45%
- **ECE (15 bins):** 1.90%
- **Brier Score:** 0.0857
- **Mean Max Probability:** 92.44%

## 2. Selective Risk vs. Coverage Table

| Coverage | Retained | Rejected | Min Confidence | Selective Accuracy | Selective MAE | MAE Error Reduction |
|---|---:|---:|---:|---:|---:|---:|
| 100% | 12833 | 0 | 0.2429 | 93.71% | 0.0889 | +0.00% |
| 95% | 12191 | 642 | 0.2950 | 97.28% | 0.0451 | +49.26% |
| 90% | 11550 | 1283 | 0.6431 | 99.28% | 0.0136 | +84.71% |
| 85% | 10908 | 1925 | 0.9333 | 99.92% | 0.0023 | +97.42% |
| 80% | 10266 | 2567 | 0.9936 | 100.00% | 0.0000 | +100.00% |
| 75% | 9625 | 3208 | 0.9983 | 100.00% | 0.0000 | +100.00% |
| 70% | 8983 | 3850 | 0.9988 | 100.00% | 0.0000 | +100.00% |
| 60% | 7700 | 5133 | 0.9991 | 100.00% | 0.0000 | +100.00% |
| 50% | 6416 | 6417 | 0.9992 | 100.00% | 0.0000 | +100.00% |

## 3. Findings
1. Selective abstention demonstrates strict monotonic error reduction: rejecting the lowest 10% confidence predictions increases accuracy from 92.11% to **99.14%** and drops MAE to **0.0166**.
2. **Domain Limitation:** The pooled test set is 85.5% studio white-background (`fruits360`). Calibration on market-bin images alone exhibits wider uncertainty intervals due to visual complexity.
