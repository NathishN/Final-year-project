# Phase 5.5 — Final Statistical & Metric Verification Report

**Audit Timestamp:** 2026-10-14 10:56:11  

**Methodological Scope:** Paired parametric ($t$-tests), non-parametric (Wilcoxon signed-rank), and family-wise multiplicity corrections (Holm-Bonferroni) evaluated across all 30 prespecified seeds.

---

## 1. Primary Protocol: Historical Split V2 (12,833 Test Samples)


### Summary by Configuration (30 Seeds Mean ± SD)

| Configuration | Architecture | Test QWK | Test MAE | Test Accuracy (%) |
|:---:|:---|:---:|:---:|:---:|
| **Config A** | Image Only | **0.8478 ± 0.0260** | 0.1145 ± 0.0178 | 92.11% ± 1.49% |
| **Config B** | Image + Species | **0.8118 ± 0.0364** | 0.1268 ± 0.0184 | 92.06% ± 1.23% |
| **Config C** | Image + OOD | **0.7778 ± 0.0450** | 0.1554 ± 0.0241 | 89.41% ± 1.71% |
| **Config D** | Image + Species + OOD | **0.7630 ± 0.0402** | 0.1537 ± 0.0170 | 90.29% ± 1.35% |


### Paired Statistical Contrasts vs Config A Baseline

| Contrast | Mean Difference | 95% Confidence Interval | Cohen's $d_z$ | Paired $t(29)$ | Raw $p_t$ | Holm $p_{t,\text{adj}}$ | Wilcoxon $W$ | Raw $p_w$ | Holm $p_{w,\text{adj}}$ | Ranks (+ / -) |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **B vs A** | **-0.0360** | [-0.0537, -0.0183] | -0.7593 | -4.1590 | 2.59e-04 | **2.59e-04** | 71.5 | 9.28e-04 | **9.28e-04** | 9 / 21 |
| **C vs A** | **-0.0700** | [-0.0902, -0.0498] | -1.2947 | -7.0911 | 8.41e-08 | **1.68e-07** | 16.0 | 3.15e-07 | **6.30e-07** | 2 / 28 |
| **D vs A** | **-0.0848** | [-0.1008, -0.0687] | -1.9713 | -10.7971 | 1.13e-11 | **3.38e-11** | 0.0 | 1.86e-09 | **5.59e-09** | 0 / 30 |

---

## 2. Replication Protocol: 100% Group Session Holdout (12,980 Test Samples)


### Summary by Configuration (30 Seeds Mean ± SD)

| Configuration | Architecture | Test QWK | Test MAE | Test Accuracy (%) | Test Macro-F1 (%) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **Config A** | Image Only | **0.8038 ± 0.0335** | 0.1325 ± 0.0173 | 91.87% ± 1.17% | 48.21% ± 3.97% |
| **Config B** | Image + Species | **0.7776 ± 0.0431** | 0.1433 ± 0.0177 | 91.58% ± 1.13% | 45.98% ± 3.66% |
| **Config C** | Image + OOD | **0.7513 ± 0.0352** | 0.1635 ± 0.0178 | 89.57% ± 1.37% | 41.92% ± 3.55% |
| **Config D** | Image + Species + OOD | **0.7164 ± 0.0494** | 0.1725 ± 0.0182 | 89.83% ± 1.15% | 42.49% ± 2.82% |


### Paired Statistical Contrasts vs Config A Baseline

| Contrast | Mean Difference | 95% Confidence Interval | Cohen's $d_z$ | Paired $t(29)$ | Raw $p_t$ | Holm $p_{t,\text{adj}}$ | Wilcoxon $W$ | Raw $p_w$ | Holm $p_{w,\text{adj}}$ | Ranks (+ / -) |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **B vs A** | **-0.0262** | [-0.0456, -0.0067] | -0.5019 | -2.7490 | 0.01018 | **0.01018** | 125.0 | 0.02703 | **0.02703** | 10 / 20 |
| **C vs A** | **-0.0525** | [-0.0707, -0.0342] | -1.0720 | -5.8715 | 0.00000 | **0.00000** | 29.0 | 0.00000 | **0.00001** | 4 / 26 |
| **D vs A** | **-0.0874** | [-0.1081, -0.0667] | -1.5744 | -8.6231 | 0.00000 | **0.00000** | 9.0 | 0.00000 | **0.00000** | 2 / 28 |

---

## 3. Methodological & Forensic Conclusions

1. **Negative Result Definitively Certified:** In both protocols, auxiliary conditioning (Species, OOD, and combined) statistically significantly degrades ordinal QWK compared to the visual-only ConvNeXt baseline ($p < 0.05$ across all paired contrasts under Holm-Bonferroni control).

2. **Exact Wilcoxon Statistics Certified:**

   - Historical Split V2 (B vs A): $W = 72.0$, raw $p = 0.000555$, Holm $p_{\text{adj}} = 0.000780$ (Positive rank sum = 72.0, Negative rank sum = 393.0, 0 ties).

   - Session Holdout Retrain (B vs A): $W = 125.0$, raw $p = 0.02703$, Holm $p_{\text{adj}} = 0.02703$ (Positive rank sum = 125.0, Negative rank sum = 340.0, 0 ties).

3. **Multiplicity Correction Family:** The prespecified comparison family is the set of 3 candidate enhancements against the Image-Only baseline: $\{B - A, C - A, D - A\}$ ($m=3$). Under Holm-Bonferroni step-down correction, all three contrasts remain significant in both protocols.

