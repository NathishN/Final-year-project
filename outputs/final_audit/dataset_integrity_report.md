# Dataset Integrity & Split Leakage Audit Report

**Date:** October 2026  
**Status:** PASSED -- Complete Reconciliation & Zero Cross-Split Leakage  

---

## 1. Master Dataset Reconciliation

- **Raw Rows in `master_dataset.csv`:** 56,038  
- **Excluded Records:** 2 (Unlabelled, corrupt, or out-of-scope images)  
- **Valid Active Records:** 56,036  
- **Unmapped Severity Labels:** 0 (100% strictly normalized)  

---

## 2. Partition Comparison: Historical (`split_v1`) vs Corrected (`split_v2`)

| Partition | Historical split_v1 | Corrected split_v2 | Delta | Grade 2 Test Coverage |
|:---|---:|---:|---:|:---:|
| **Train** | 39,909 | 39,790 | -119 | 208 |
| **Val** | 3,413 | 3,413 | 0 | 196 |
| **Test** | 12,714 | 12,833 | +119 | **119** |
| **Total** | 56,036 | 56,036 | 0 | 523 |

> **Key Resolution:** In historical `split_v1`, Grade 2 had exactly **0 test samples**, preventing genuine 5-grade evaluation. In `split_v2`, session `20200730` (119 Banana Grade 2 images) was reallocated to test, establishing the first valid 5-grade ordinal test set.

---

## 3. Grade Representation in Corrected `split_v2`

| Severity Grade | Train Count | Val Count | Test Count | Total Count | Test Ratio (%) |
|:---|---:|---:|---:|---:|---:|
| **Healthy (0)** | 36,414 | 1,037 | 11,605 | 49,056 | 23.66% |
| **Mild (1)** | 846 | 494 | 88 | 1,428 | 6.16% |
| **Moderate (2)** | 208 | 196 | 119 | 523 | 22.75% |
| **Severe (3)** | 1,606 | 1,268 | 868 | 3,742 | 23.2% |
| **Rotten (4)** | 716 | 418 | 153 | 1,287 | 11.89% |

---

## 4. Cross-Split Leakage Verification

- **Path-Level Leakage (Train ∩ Test):** **0** (Verified)
- **Path-Level Leakage (Train ∩ Val):** **0** (Verified)
- **Path-Level Leakage (Val ∩ Test):** **0** (Verified)
- **Grade-2 Session Leakage (Train ∩ Test):** **0** (Verified; session `20200730` does not exist in train or val)

---

## 5. Domain Breakdown in Test Set (`split_v2`)

| Acquisition Domain | Test Sample Count | Pct of Test (%) | Severity Grade Coverage |
|:---|---:|---:|:---|
| `fruits360` (Studio Turntable) | 10,974 | 85.51% | Grade 0 Only |
| `real_quality` (Market Bins) | 1,480 | 11.53% | Grades 0, 2, 3 |
| `real_severity` (Field Images) | 312 | 2.43% | Grades 0, 1, 4 |
| `real_grade` (Grading Station) | 67 | 0.52% | Grades 0, 1, 3 |

**Conclusion:** Dataset integrity verified. No cross-split contamination detected.