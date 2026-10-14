# Confirmatory Ordinal-Grading Experiment Report

**Date:** October 2026  
**Protocol:** Frozen `split_v2` (12,833 test samples; 119 Grade 2 test samples)  
**Scope:** 4 Configurations × 30 Prespecified Seeds = 120 Independent Runs  

---

## 1. Primary Hypothesis Evaluation (Configuration B vs Configuration A)

> **Prespecified Primary Comparison:** Does Species Conditioning (Config B: Image + Species) significantly improve Quadratic Weighted Kappa (QWK) over Image-Only (Config A)?

- **Config A (Image Only) QWK:** 0.8478 ± 0.0260  
- **Config B (Image + Species) QWK:** 0.8118 ± 0.0364  
- **Mean Paired Difference (ΔQWK):** -0.0360  
- **95% Confidence Interval:** [-0.0537, -0.0183]  
- **Paired Two-Sided t-Test:** t(29) = -4.1590, **p = 0.000259**  
- **Wilcoxon Signed-Rank Test:** **p = 0.000928**  
- **Paired Effect Size (Cohen's d_z):** -0.7593  
- **Hypothesis Verdict:** **SIGNIFICANT DEGRADATION**  

---

## 2. Aggregated 4-Configuration Benchmark ($N=30$ Seeds per Config)

| Config | Architecture / Inputs | QWK Mean ± Std | MAE Mean ± Std | Accuracy (%) | OBO Acc (%) | Macro-F1 (%) | Grade 2 Recall (%) | Pred G2 Count |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Image Only | **0.8478 ± 0.0260** | 0.1145 ± 0.0178 | 92.11% | 97.30% | 49.48% | 7.4% | 481.1 |
| **B** | Image + Species | **0.8118 ± 0.0364** | 0.1268 ± 0.0184 | 92.06% | 96.56% | 49.35% | 21.5% | 415.4 |
| **C** | Image + OOD | **0.7778 ± 0.0450** | 0.1554 ± 0.0241 | 89.41% | 95.61% | 42.39% | 6.6% | 537.0 |
| **D** | Image + Species + OOD | **0.7630 ± 0.0402** | 0.1536 ± 0.0170 | 90.29% | 95.34% | 44.39% | 12.7% | 440.1 |

---

## 3. Pairwise Statistical Contrasts

| Contrast | Comparison Role | Mean ΔQWK | 95% Confidence Interval | t-Statistic | p-Value (t-Test) | Cohen's d_z | Statistical Verdict |
|:---|:---:|---:|:---:|---:|---:|---:|:---|
| **B vs A (Primary: Species)** | Primary | -0.0360 | [-0.0537, -0.0183] | -4.1590 | 0.000259 | -0.7593 | **SIGNIFICANT DEGRADATION** |
| **C vs A (Secondary: OOD)** | Secondary | -0.0700 | [-0.0902, -0.0498] | -7.0911 | 0.000000 | -1.2947 | **SIGNIFICANT DEGRADATION** |
| **D vs A (Secondary: Full vs Image)** | Secondary | -0.0848 | [-0.1008, -0.0687] | -10.7971 | 0.000000 | -1.9713 | **SIGNIFICANT DEGRADATION** |
| **D vs B (Secondary: OOD Incremental)** | Secondary | -0.0488 | [-0.0678, -0.0298] | -5.2530 | 0.000013 | -0.9591 | **SIGNIFICANT DEGRADATION** |

---

## 4. Scientific Discussion of Trade-Offs

1. **Aggregate QWK vs Minority Class Recall:**  
   While Configuration A achieves superior overall QWK (0.8478 vs 0.8118) and lower MAE (0.1145 vs 0.1268),    Configuration B achieves higher recall on the rare Grade 2 class (21.5% vs 7.4%).    This indicates that species conditioning acts as a regularizer toward minority-grade boundaries,    but introduces noise on dominant Grade 0 samples.

2. **OOD Feature Conditioning Degradation:**  
   Adding Mahalanobis OOD distance features (Config C and Config D) significantly degrades QWK across all 30 seeds    ($\Delta = -0.0700, p = 8.4 \times 10^{-8}$). Open-set distance representations do not correlate with ordinal severity    and should NOT be concatenated into the severity grading bottleneck.