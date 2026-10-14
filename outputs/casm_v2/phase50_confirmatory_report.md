# Phase 5.0 -- Stage 5: Confirmatory Ordinal-Grading Report

**Date:** October 2026  
**Evaluation Protocol:** Frozen `split_v2` (12,833 test samples; 119 Grade 2 test samples)  
**Seeds:** 30 Prespecified Seeds (All 30 executed deterministically; 0 dropped)  
**Backbone:** Frozen ConvNeXt-Tiny embeddings with MonotonicCORALHead & Effective Number loss  

---

## 1. Primary Hypothesis Evaluation (Configuration B vs Configuration A)

> **Prespecified Primary Comparison:** Does Species Conditioning (Config B: Image + Species) significantly improve Quadratic Weighted Kappa (QWK) over Image-Only (Config A)?

- **Config A (Image Only) QWK:** 0.8478 ± 0.0260  
- **Config B (Image + Species) QWK:** 0.8118 ± 0.0364  
- **Paired Mean Difference (ΔQWK):** -0.0360  
- **95% Confidence Interval:** [-0.0537, -0.0183]  
- **Paired t-test:** t(29) = -4.1590, **p = 0.000259**  
- **Wilcoxon Signed-Rank Test:** W = 71.5, **p = 0.000928**  
- **Paired Effect Size (Cohen's d_z):** -0.7593  
- **Primary Hypothesis Verdict:** **SIGNIFICANT DEGRADATION**  

---

## 2. Comprehensive 4-Configuration Benchmark

| Config | Name | QWK | MAE | Accuracy (%) | OBO Acc (%) | Macro-F1 (%) | Grade 2 Recall (%) | Pred G2 Count |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Image Only | 0.8478 ± 0.0260 | 0.1145 ± 0.0178 | 92.11% | 97.30% | 49.48% | 7.4% | 481.1 |
| **B** | Image + Species | 0.8118 ± 0.0364 | 0.1268 ± 0.0184 | 92.06% | 96.56% | 49.35% | 21.5% | 415.4 |
| **C** | Image + OOD | 0.7778 ± 0.0450 | 0.1554 ± 0.0241 | 89.41% | 95.61% | 42.39% | 6.6% | 537.0 |
| **D** | Image + Species + OOD | 0.7630 ± 0.0402 | 0.1536 ± 0.0170 | 90.29% | 95.34% | 44.39% | 12.7% | 440.1 |

---

## 3. Pairwise Statistical Contrasts

| Contrast | Comparison Type | Mean ΔQWK | 95% CI | t-stat | p-value (t-test) | Cohen's d_z | Verdict |
|:---|:---:|---:|:---:|---:|---:|---:|:---|
| **B vs A (Primary: Species Effect)** | Primary | -0.0360 | [-0.0537, -0.0183] | -4.1590 | 0.000259 | -0.7593 | **SIGNIFICANT DEGRADATION** |
| **C vs A (Secondary: OOD Effect)** | Secondary | -0.0700 | [-0.0902, -0.0498] | -7.0911 | 0.000000 | -1.2947 | **SIGNIFICANT DEGRADATION** |
| **D vs A (Secondary: Full vs Image)** | Secondary | -0.0848 | [-0.1008, -0.0687] | -10.7971 | 0.000000 | -1.9713 | **SIGNIFICANT DEGRADATION** |
| **D vs B (Secondary: OOD Incremental)** | Secondary | -0.0488 | [-0.0678, -0.0298] | -5.2530 | 0.000013 | -0.9591 | **SIGNIFICANT DEGRADATION** |

---

## 4. Key Scientific Findings & Discussion

1. **Full 5-Grade Evaluation Validation:**  
   For the first time in this research project, all five severity grades (0, 1, 2, 3, 4) were evaluated on test data.
   Grade 2 test samples (n=119) were successfully evaluated without cross-split leakage.

2. **Species Conditioning Empirical Evidence:**  
   The primary hypothesis test yielded ΔQWK = -0.0360 (p = 0.000259). 
   This provides direct, rigorous statistical evidence on whether species embeddings enhance ordinal severity grading.

3. **Monotonicity & Stability Guarantees:**  
   Under MonotonicCORALHead, 100% of the 120 confirmatory runs converged cleanly with 0 NaN/Inf values,    strictly valid class probabilities, and bounded minority-grade predictions.

4. **Artifact Audit Readiness:**  
   All 120 model checkpoints and raw per-sample prediction tensors are persisted on disk for independent verification.

---

## 5. Deliverable Checklist
- `phase50_checkpoints/`: 120 model checkpoints saved
- `phase50_per_seed_predictions/`: 120 per-seed prediction `.pt` files saved
- `phase50_confirmatory_results.csv`: Complete 120-row result table saved
- `phase50_confirmatory_statistics.csv`: Exact parametric and non-parametric tests saved
- `phase50_confirmatory_report.md`: Confirmatory summary saved