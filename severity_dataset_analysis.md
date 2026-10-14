# Severity Dataset Analysis & Taxonomy Harmonization Report

**Paper Title:** *Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration*  
**Phase:** Phase 4 — Context-Aware Severity Modeling (CASM)  
**Date:** October 2026  

---

## 1. Executive Summary

This report provides a rigorous dataset analysis and taxonomy harmonization for **Phase 4: Context-Aware Severity Modeling (CASM)**. Multi-fruit quality grading systems operating across heterogeneous agricultural domains face significant label fragmentation, class imbalance, and domain shift. In this work, we analyze **56,038 images** across 5 major economic fruit species (*Apple, Banana, Mango, Orange, Tomato*) collected from 4 distinct acquisition domains:
1. `fruits360`: Synthetic/laboratory controlled isolated fruit split.
2. `real_quality`: Real-world market quality tiers (*Good, Mixed, Bad*).
3. `real_severity`: Biological rot severity annotations (*Fresh, Mild, Rotten*).
4. `real_grade`: Commercial export grade standards (*1st Grade, 2nd Grade, 3rd Grade*).

To enable rank-consistent ordinal severity regression, we design a unified **5-Level Ordinal Severity Taxonomy** (\(S \in \{0, 1, 2, 3, 4\}\)):
- **Rank 0 (Healthy)**: Pristine fruit without observable physical, fungal, or surface blemishes.
- **Rank 1 (Mild Defect)**: Minor surface blemishes, slight discoloration, or 2nd grade commercial flaws.
- **Rank 2 (Moderate Defect)**: Heterogeneous surface degradation, moderate physiological flaws, or mixed quality tier.
- **Rank 3 (Severe Defect)**: Major physical injury, structural damage, 3rd grade processing tier, or severe quality degradation.
- **Rank 4 (Rotten)**: Advanced biological decay, fungal colonization, active necrosis, and total commercial loss.

---

## 2. Dataset Composition & Domain Statistics

The dataset comprises **56,036 valid images** (with 2 images excluded due to insufficient sample size in `Banana_3rd_grade`).

### Table 1: Cross-Domain Sample Distribution by Fruit Species

| Fruit Species | Fruits-360 | Real Quality | Real Severity | Real Grade | Total Samples | Domain Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple** | 22,077 | 2,403 | 0 | 482 | 24,962 | 44.54% |
| **Banana** | 2,562 | 2,485 | 612 | 482 | 6,141 | 10.96% |
| **Mango** | 1,224 | 0 | 0 | 618 | 1,842 | 3.29% |
| **Orange** | 4,427 | 2,500 | 0 | 0 | 6,927 | 12.36% |
| **Tomato** | 13,672 | 0 | 1,990 | 502 | 16,164 | 28.85% |
| **Total** | **43,962** | **7,388** | **2,602** | **2,084** | **56,036** | **100.00%** |

---

## 3. Unified Ordinal Severity Distribution

### Table 2: 5-Class Ordinal Severity Breakdown across Splits

| Severity Rank | Description | Train Split | Val Split | Test Split | Overall Count | Overall Ratio (%) | Real-World Count | Real-World Ratio (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Healthy | 36,414 | 1,037 | 11,605 | 49,056 | 87.54% | 5,094 | 42.19% |
| **1** | Mild Defect | 846 | 494 | 88 | 1,428 | 2.55% | 1,428 | 11.83% |
| **2** | Moderate Defect | 327 | 196 | 0 | 523 | 0.93% | 523 | 4.33% |
| **3** | Severe Defect | 1,606 | 1,268 | 868 | 3,742 | 6.68% | 3,742 | 30.99% |
| **4** | Rotten | 716 | 418 | 153 | 1,287 | 2.30% | 1,287 | 10.66% |
| **Total** | | **39,909** | **3,413** | **12,714** | **56,036** | **100.00%** | **12,074** | **100.00%** |

---

## 4. Class Imbalance & Inter-Domain Conflict Analysis

1. **Class Imbalance Ratio**:
   - Overall dataset: Max-to-min class imbalance ratio is \(49,056 / 523 = 93.8:1\).
   - Real-world sub-cohort (12,074 samples): Max-to-min ratio is \(5,094 / 523 = 9.74:1\).
   - *Mitigation*: We implement Class-Weighted CORAL Loss, Focal Ordinal Loss, and balanced sample weighting during context MLP fusion training.

2. **Domain Shift & Background Bias**:
   - `fruits360` samples exhibit homogeneous white backgrounds with centered studio lighting, presenting zero environmental noise.
   - `real_grade`, `real_quality`, and `real_severity` domains feature natural orchard, packing line, and consumer market backgrounds with complex specular reflections and shadows.
   - *Mitigation*: Pre-extracted 768-D ConvNeXt-Tiny embeddings are fused with contextual vectors (species, grade, OOD score, uncertainty entropy) to ensure severity estimation relies on visual-context interaction rather than background features.

3. **Annotation Quality & Inter-Domain Mapping Consistency**:
   - Mapping rules are rigorously preserved in `severity_mapping.json`.
   - `Banana_3rd_grade` contained only \(n=2\) images and was systematically excluded to maintain session-stratified split integrity.

---

## 5. Conclusion & Transition to Phase 4 Modeling

The established 5-class ordinal taxonomy provides a mathematically sound target representation for Phase 4 Context-Aware Severity Modeling (CASM). The inclusion of multi-source domain context directly addresses the limitations of visual-only classification, serving as the benchmark for context ablation studies.
