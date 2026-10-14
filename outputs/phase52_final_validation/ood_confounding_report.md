# Phase 5.2 — OOD Detector Evaluation & Domain Confounding Audit

## 1. Detector Comparison on Real Open-Set Benchmark

5,000 real images of unseen fruit species (Cherry, Kiwi, Peach, Pineapple, Plum; 1,000 each) were evaluated against 12,833 in-distribution test samples.

| Detector | AUROC | AUPR (OOD) | FPR @ 95% TPR | 95% Confidence Interval |
|---|---:|---:|---:|:---:|
| **Mahalanobis** | 84.22% | 53.28% | 38.39% | [83.7%, 84.8%] |
| **Energy** | 81.99% | 49.96% | 41.38% | [81.3%, 82.6%] |
| **MSP** | 74.26% | 43.16% | 47.85% | [73.6%, 74.9%] |

## 2. Acquisition-Domain Inversion (The Confounding Phenomenon)

| In-Distribution Subset | Sample Count | Mahalanobis AUROC | Energy AUROC | MSP AUROC |
|---|---:|---:|---:|---:|
| `fruits360` | 10974 | 93.26% | 79.99% | 75.21% |
| `real_grade` | 67 | 18.81% | 98.89% | 96.57% |
| `real_quality` | 1480 | 18.08% | 92.49% | 61.74% |
| `real_severity` | 312 | 94.10% | 98.67% | 95.30% |

### Scientific Assessment of Confounding:
1. **The Inversion:** When ID images originate from `fruits360` (clean studio white background), Mahalanobis AUROC is **93.26%**. However, when ID images originate from `real_quality` (market bins with complex visual backgrounds), Mahalanobis AUROC collapses to **18.08%** (worse than random guessing, indicating an inverted feature distribution).
2. **Detector Resilience:** In sharp contrast, the **Energy score** achieves **92.49%** AUROC on `real_quality` and **98.89%** on `real_grade`, demonstrating that logit-space energy is far more robust to domain shifts than feature-space Mahalanobis distance.
3. **Causal Attribution Status: UNRESOLVED:** Although studio-versus-market visual background is an obvious candidate confounder, the raw photographic dataset does not contain pixel-level segmentation masks separating foreground fruits from background surfaces. Therefore, claiming that 'background caused the inversion' is a correlational hypothesis, not a proven causal mechanism. This distinction is strictly maintained in the manuscript.
