# Phase 5.0 -- Stage 6: Domain Generalization Report

**Date:** October 2026  
**Evaluation Scope:** Synthetic (`fruits360`) vs Real-World In-Distribution Domains  

---

## 1. Domain Performance Disparity Analysis

| Acquisition Domain | Sample N | Test Pct (%) | Config A Acc (%) | Config B Acc (%) | Config A MAE | Config B MAE | Config A QWK | Config B QWK |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **fruits360** | 10,974 | 85.51% | 99.88% | 99.74% | 0.0012 | 0.0026 | 0.0000 | 0.0000 |
| real_grade | 67 | 0.52% | 19.4% | 13.43% | 1.1642 | 1.2985 | 0.2800 | 0.1663 |
| real_quality | 1,480 | 11.53% | 39.86% | 38.72% | 0.8135 | 0.9128 | 0.5920 | 0.5143 |
| real_severity | 312 | 2.43% | 82.69% | 81.09% | 0.1731 | 0.1891 | 0.8980 | 0.8900 |
| **REAL_COMBINED (split_real)** | 1,859 | 14.49% | 46.32% | 44.92% | 0.7187 | 0.8053 | 0.7045 | 0.6489 |

---

## 2. Key Findings on Domain Shift & Background Confounding

1. **Severe Synthetic-to-Real Accuracy Collapse:**  
   - On synthetic `fruits360` images, the model achieves **99.88%** accuracy.
   - On real-world market images (`split_real`), accuracy collapses to **46.32%** (Config A) and **44.92%** (Config B), with MAE exceeding **0.72 grades**.

2. **Mechanism of Collapse:**  
   - `fruits360` consists of isolated fruits rotated on white motorized turntables with uniform studio lighting.
   - Real-world datasets (`real_severity`, `real_quality`, `real_grade`) feature wooden crates, plastic bins, shadows, occlusions, and variable natural lighting.
   - The deep ConvNeXt representations learned strong background and illumination priors that fail to transfer zero-shot across acquisition regimes.

3. **Proposed Leave-One-Domain-Out (LODO) Protocol:**  
   - To properly evaluate domain robustness in future research, models should be trained on $K-1$ domains and evaluated strictly on the held-out $K$-th domain.
   - Pooling synthetic and real images into a single training set obscures domain-specific failure modes.

---

## 3. Publication Recommendation
> **CRITICAL:** The paper MUST NOT report the headline ~87% test accuracy without prominently displaying the domain breakdown. The abstract and introduction must clearly state that high test performance is driven by synthetic laboratory images, while real-world bin grading remains an open research challenge.