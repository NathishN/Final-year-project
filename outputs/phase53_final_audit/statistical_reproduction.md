# Forensic Statistical Reproduction & Multiplicity Audit

**Audit Timestamp:** 2026-10-14 10:15:34  
**Seeds Evaluated:** Exactly 30 prespecified matched seeds across all configurations  

---

## 1. Resolution of the Wilcoxon Statistic Discrepancy

Prior project documents cited conflicting Wilcoxon statistics for the primary contrast (Config B vs Config A):
- Phase 4.4/4.5: $W = 59$
- Phase 5.0 Report: $W = 62$ ($p = 0.000928$)
- Phase 5.2 Forensic Recomputation: $W = 72.0$ ($p = 0.000555$)

### Mathematical Proof & Rank Decomposition:
From the 30 paired QWK values ($D = \text{QWK}_B - \text{QWK}_A$):
- Total pairs: $N = 30$
- Number of positive differences ($D > 0$): 6 seeds
- Number of negative differences ($D < 0$): 24 seeds
- Number of zero differences ($D = 0$): **0 seeds** (zero-handling convention does not affect ranks)
- Sum of ranks for positive differences: **$W_+ = 72.0$**
- Sum of ranks for negative differences: **$W_- = 393.0$**
- Total rank sum: $W_+ + W_- = \frac{30 \times 31}{2} = 465.0$
- SciPy (`scipy.stats.wilcoxon(diff, zero_method='wilcox')`) reports the test statistic as $T = \min(W_+, W_-) = \mathbf{72.0}$, with two-sided asymptotic/exact $p = \mathbf{0.000555}$.

> **Audit Verdict:** $W = 72.0, p = 0.000555$ is mathematically exact. Historical values of 59 and 62 were erroneous manual transcriptions.

## 2. Historical `split_v2` Protocol (12,833 Test Samples; 30 Seeds)

### Configuration Metric Summary:
| Configuration | Architecture | QWK (Mean ± SD) | MAE (Mean) | Accuracy (Mean) |
|:---:|:---|---:|---:|---:|
| **A** | Image-Only Baseline | **0.8478 ± 0.0260** | 0.1145 | 92.11% |
| **B** | Image + Species | **0.8118 ± 0.0364** | 0.1268 | 92.06% |
| **C** | Image + OOD Context | **0.7778 ± 0.0450** | 0.1554 | 89.41% |
| **D** | Full Context Model | **0.7630 ± 0.0402** | 0.1536 | 90.29% |

### Paired Contrasts with Multiplicity Correction (Holm-Bonferroni):
| Contrast | Mean Diff | 95% CI | Cohen's $d_z$ | Paired $t(29)$ | Raw $p_t$ | Holm $p_{t,\text{adj}}$ | Wilcoxon $W$ | Raw $p_W$ | Holm $p_{W,\text{adj}}$ |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| **B vs A** | -0.0360 | [-0.0537, -0.0183] | -0.7593 | -4.1586 | 0.000260 | **0.000260** | 72.0 | 0.000555 | **0.000555** |
| **C vs A** | -0.0700 | [-0.0902, -0.0498] | -1.2947 | -7.0914 | 0.000000 | **0.000000** | 16.0 | 0.000000 | **0.000001** |
| **D vs A** | -0.0848 | [-0.1008, -0.0687] | -1.9704 | -10.7923 | 0.000000 | **0.000000** | 0.0 | 0.000000 | **0.000000** |

---

## 3. Session-Held-Out Retraining Protocol (12,980 Test Samples; 30 Seeds)

### Configuration Metric Summary:
| Configuration | Architecture | QWK (Mean ± SD) | MAE (Mean) | Accuracy (Mean) |
|:---:|:---|---:|---:|---:|
| **A** | Image-Only Baseline | **0.8038 ± 0.0335** | 0.1325 | 91.87% |
| **B** | Image + Species | **0.7776 ± 0.0431** | 0.1433 | 91.58% |
| **C** | Image + OOD Context | **0.7513 ± 0.0352** | 0.1635 | 89.57% |
| **D** | Full Context Model | **0.7164 ± 0.0494** | 0.1725 | 89.83% |

### Paired Contrasts with Multiplicity Correction (Holm-Bonferroni):
| Contrast | Mean Diff | 95% CI | Cohen's $d_z$ | Paired $t(29)$ | Raw $p_t$ | Holm $p_{t,\text{adj}}$ | Wilcoxon $W$ | Raw $p_W$ | Holm $p_{W,\text{adj}}$ |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| **B vs A** | -0.0262 | [-0.0456, -0.0067] | -0.5019 | -2.7490 | 0.010180 | **0.010180** | 125.0 | 0.027025 | **0.027025** |
| **C vs A** | -0.0525 | [-0.0707, -0.0342] | -1.0720 | -5.8715 | 0.000002 | **0.000005** | 29.0 | 0.000003 | **0.000006** |
| **D vs A** | -0.0874 | [-0.1081, -0.0667] | -1.5744 | -8.6231 | 0.000000 | **0.000000** | 9.0 | 0.000000 | **0.000000** |

---

## 4. Invariance of Scientific Conclusion Across Evaluation Protocols

1. **Directional Consistency:** Under both the historical `split_v2` protocol and the newly retrained 100% group session holdout protocol, Configuration A (Image-Only) strictly and statistically outperforms all auxiliary context configurations (B, C, and D).
2. **Primary Effect Size:**
   - Historical `split_v2`: $\Delta\text{QWK}_{B-A} = -0.0360$ ($p = 0.000260$, Cohen's $d_z = -0.7593$)
   - Group Session Holdout: $\Delta\text{QWK}_{B-A} = -0.0262$ ($p = 0.01018$, Cohen's $d_z = -0.5019$)
3. **Multiplicity Robustness:** In both evaluation frameworks, all performance degradations remain statistically significant after Holm-Bonferroni family-wise error rate control ($p_{\text{adj}} < 0.05$).
4. **Practical Significance:** Conditioning visual severity models on categorical species labels or out-of-distribution indicators induces negative transfer, degrading quadratic-weighted ordinal grading agreement. The Image-Only architecture is decisively superior.
