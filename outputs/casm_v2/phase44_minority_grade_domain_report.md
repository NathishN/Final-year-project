# Phase 4.4 Task F — Minority Grade Imbalance & Domain Confounding Audit

**Auditors:** Statistical ML Researcher & Agricultural AI Specialist  
**Date:** October 2026  
**Repository:** `C:\fruit_dataset`  
**Primary Artifacts:** `outputs/casm_v2/dataset_statistics.json`, `outputs/casm_v2/dataset_audit.md`

---

## 1. Executive Summary

This report audits the severity label distributions, partition structure, and domain provenance across the 56,036 fruit images in the dataset.

### Key Discoveries:
1. **Extreme Class Imbalance:** Grade 0 ("Healthy") constitutes **$87.54\%$** of the entire dataset ($49,056 / 56,036$), while Grade 2 ("Moderate Defect") accounts for only **$0.93\%$** ($523$ samples).
2. **Total Absence of Grade 2 from Test Partition:** The test set ($N=12,714$) contains **zero instances of Grade 2**. Any test evaluation operates over an effective 4-class taxonomy, distorting standard 5-class macro metrics.
3. **Severe Domain-Severity Confounding:** The synthetic/laboratory dataset `fruits360` accounts for $78.45\%$ of all samples and consists exclusively of pristine Grade 0 fruit on solid white backgrounds. Consequently, background cleanliness is an accidental visual shortcut for Grade 0.

---

## 2. Severity Grade Distribution Across Splits

### Table 1: Cross-Split Sample Counts and Class Ratios

| Grade | Label Semantics | Train Split | Val Split | Test Split | Total | Global Ratio | Test Split Ratio |
|:---:|---|---:|---:|---:|---:|---:|---:|
| **0** | Healthy | 36,414 | 1,037 | **11,605** | 49,056 | 87.54% | **91.28%** |
| **1** | Mild Defect | 846 | 494 | **88** | 1,428 | 2.55% | **0.69%** |
| **2** | Moderate Defect | 327 | 196 | **0** | 523 | 0.93% | **0.00%** |
| **3** | Severe Defect | 1,606 | 1,268 | **868** | 3,742 | 6.68% | **6.83%** |
| **4** | Rotten | 716 | 418 | **153** | 1,287 | 2.30% | **1.20%** |
| **Total** | | **39,909** | **3,413** | **12,714** | **56,036** | **100.00%** | **100.00%** |

### Critical Diagnostic: The Missing Grade 2
In the test set:
- Healthy: $11,605$
- Defect grades (1, 3, 4): $1,109$ combined
- Grade 2: **0**

Because Grade 2 is completely absent from the test set:
1. The model's learned threshold $b_2$ separating Grade 1 and Grade 2, and threshold $b_3$ separating Grade 2 and Grade 3, are **never tested in isolation on the test partition**.
2. Any prediction of Grade 2 on the test set is penalized as a false positive with zero potential true positives.
3. When computing Scikit-learn's unweighted `f1_score(average='macro', zero_division=0)`, Class 2 receives an exact score of $0.0000$. The maximum attainable Macro-F1 on the test set is therefore capped at:
   $$\text{Max Macro-F1} = \frac{1 + 1 + 0 + 1 + 1}{5} = \mathbf{0.8000}$$

---

## 3. Domain Composition & Visual Confounding

The 56,036 images originate from four acquisition domains with fundamentally distinct photographic conditions:

### Table 2: Domain Breakdown and Severity Profile

| Domain | Total Images | Dataset % | Photographic Environment | Severity Representation |
|---|---:|---:|---|---|
| `fruits360` | 43,962 | 78.45% | Turntable rig, pure white background | **100% Grade 0 (Pristine)** |
| `real_quality` | 7,388 | 13.18% | Retail market lighting, natural shelves | Grades 0, 1, 3, 4 |
| `real_severity` | 2,602 | 4.64% | Inoculated decay series, varied backgrounds | Grades 1, 2, 3, 4 |
| `real_grade` | 2,084 | 3.72% | Commercial sorting line conveyor | Grades 1, 2, 3 |

### The Shortcut Learning Risk
Because `fruits360` represents $89.6\%$ of all Grade 0 images in the dataset and $78.45\%$ of all images overall:
- Deep convolutional backbones (including ConvNeXt-Tiny) can achieve $>86\%$ accuracy simply by identifying whether an image contains a segmented white studio background versus a complex retail background.
- This creates severe vulnerability to domain shift: a pristine fruit photographed in a realistic orchard or grocery bin may be misclassified as defective simply because its background is complex.

---

## 4. Methodological Remediation for Paper Submission

To present these findings with complete scientific integrity in an IEEE Transactions manuscript, the following disclosures and analyses are mandatory:

1. **Explicit Test Set Specification:**
   - Clearly define the test set as an active 4-class evaluation set: $\mathcal{Y}_{\text{test}} \in \{0, 1, 3, 4\}$.
   - Report both the standard 5-class Macro-F1 ($\sim 30\%$) and the active 4-class Macro-F1 ($\sim 37.5\%$).
2. **Domain-Stratified Robustness Reporting:**
   - In addition to aggregate test accuracy, report performance broken down by acquisition domain.
   - Specifically evaluate accuracy on the `real_quality` and `real_severity` sub-cohorts, which contain challenging real-world lighting and backgrounds.
3. **Ordinal Metrics Over Nominal Classification:**
   - Emphasize Quadratic Weighted Kappa (QWK) and Off-By-One (OBO) accuracy over Top-1 Accuracy.
   - QWK and OBO inherently penalize severity mispredictions proportionally to rank distance, mitigating the binary distortion caused by the massive Grade 0 majority class.
