# Phase 4.5 — Dataset Integrity and Grade Coverage Audit Report

**Auditors:** Senior Machine Learning Research Engineer, Computer Vision Scientist, Statistical Auditor  
**Date:** October 2026  
**Repository:** `C:\fruit_dataset`  
**Primary Deliverables:**
- `outputs/casm_v2/phase45_dataset_integrity.json`
- `outputs/casm_v2/phase45_dataset_integrity_report.md`

---

## 1. Executive Summary

A comprehensive, ground-truth audit of `master_dataset.csv`, `severity_mapping.json`, raw image folders, and image content hashes was performed. All data counts were derived directly from raw files rather than cached summaries.

### Principal Findings:
1. **Total Dataset Census:** Exactly **56,038** total records exist in `master_dataset.csv`. Exactly 2 samples (`Banana_3rd_grade`) are marked as `EXCLUDED`, leaving **56,036** active experimental images:
   - **Train Split:** 39,909 images (71.22%)
   - **Validation Split:** 3,413 images (6.09%)
   - **Test Split:** 12,714 images (22.69%)
2. **Path and Content Isolation:**
   - Path leakage across splits: **0** images.
   - MD5 content hash duplicates across splits (audited over all 12,076 non-Fruits360 images): **0** duplicates.
   - File path existence on disk: **56,036 / 56,036 (100.0%) verified**.
   - Metadata vs folder hierarchy conflicts: **0** conflicts.
3. **The "Missing Grade 2" Root Cause (Task 2.2):** Grade 2 ("Moderate Defect") has **exactly 0 samples in the test split** ($N_{\text{test}, 2} = 0$). This was definitively traced to an **unstratified session allocation flaw in `generate_master_csv.py`**. The session-stratified split grouped images by `(fruit, domain)` rather than `(fruit, domain, severity_grade)`. Because Grade 2 (`mixed` quality) only exists across three calendar sessions (`20200728`, `20200729`, `20200730`), the pseudorandom shuffle at seed 42 allocated two sessions to `train` ($n=327$) and one session to `val` ($n=196$), allocating **zero sessions to `test`**.
4. **Domain Confounding and Class Imbalance:**
   - Grade 0 comprises **87.54%** of the entire dataset ($49,056 / 56,036$) and **91.28%** of the test set ($11,605 / 12,714$).
   - `fruits360` accounts for **78.45%** of all images ($43,962 / 56,036$) and **86.31%** of the test set ($10,974 / 12,714$).
   - **Every single `fruits360` image is mapped to Grade 0.**
   - No individual real-world domain contains all 5 severity grades. Grade 2 is exclusively supplied by `real_quality`, Grade 4 is exclusively supplied by `real_severity`, and `fruits360` exclusively supplies Grade 0.

---

## 2. Reconstructed Complete Label Distribution (Task 2.1)

The complete distribution was independently reconstructed from the active master table:

### Table 1: Complete Partition and Severity Grade Census
| Severity Grade | Clinical / Commercial Rank | Train Split | Validation Split | Test Split | Total Active | Percentage |
|:---:|---|---:|---:|---:|---:|---:|
| **0** | Healthy | 36,414 | 1,037 | 11,605 | **49,056** | 87.54% |
| **1** | Mild Defect | 846 | 494 | 88 | **1,428** | 2.55% |
| **2** | Moderate Defect | 327 | 196 | **0** | **523** | 0.93% |
| **3** | Severe Defect | 1,606 | 1,268 | 868 | **3,742** | 6.68% |
| **4** | Rotten | 716 | 418 | 153 | **1,287** | 2.30% |
| **Total** | | **39,909** | **3,413** | **12,714** | **56,036** | **100.00%** |

*(Note: 2 excluded Banana images in Banana_3rd_grade have severity 3, making total CSV rows 56,038).*

---

### Table 2: Acquisition Domain Census by Split
| Acquisition Domain | Sensor / Source Environment | Train Split | Val Split | Test Split | Total Images | Total Share |
|---|---|---:|---:|---:|---:|---:|
| `fruits360` | Laboratory turntable, white background | 32,988 | 0 | 10,974 | **43,962** | 78.45% |
| `real_quality` | Variable lighting, market quality bins | 4,163 | 1,864 | 1,361 | **7,388** | 13.18% |
| `real_severity` | Mobile phone, disease progression stages | 1,515 | 775 | 312 | **2,602** | 4.64% |
| `real_grade` | Commercial grading standards | 1,243 | 774 | 67 | **2,084** | 3.72% |
| **Total** | | **39,909** | **3,413** | **12,714** | **56,036** | **100.00%** |

**Observation on `fruits360` Validation Split:** Notice that `fruits360` has **0 validation images**. In `generate_master_csv.py`, the official Fruits-360 `TRAIN360` folders were mapped directly to `train` ($32,988$) and `TEST360` folders were mapped directly to `test` ($10,974$). Consequently, the validation split ($n=3,413$) is composed **100% of real-world images**, whereas the test split is **86.31% synthetic laboratory images**. This explains why validation MAE and test MAE exhibit severe distribution shifts during training!

---

### Table 3: Joint Distribution: Severity Grade × Species (All 56,036 Images)
| Severity Grade | Apple | Banana | Mango | Orange | Tomato | Total |
|:---:|---:|---:|---:|---:|---:|---:|
| **0 (Healthy)** | 23,424 | 4,097 | 1,475 | 5,643 | 14,417 | **49,056** |
| **1 (Mild)** | 172 | 335 | 250 | 0 | 671 | **1,428** |
| **2 (Moderate)** | 113 | 285 | 0 | 125 | 0 | **523** |
| **3 (Severe)** | 1,253 | 1,087 | 117 | 1,159 | 126 | **3,742** |
| **4 (Rotten)** | 0 | 337 | 0 | 0 | 950 | **1,287** |
| **Total** | **24,962** | **6,141** | **1,842** | **6,927** | **16,164** | **56,036** |

**Critical Taxonomic Gaps:**
- Orange has **0 samples** of Grade 1 and **0 samples** of Grade 4.
- Apple has **0 samples** of Grade 4 (Rotten).
- Mango has **0 samples** of Grade 2 and **0 samples** of Grade 4.
- Tomato has **0 samples** of Grade 2.
- Only **Banana** possesses non-zero sample representation across all 5 severity grades.

---

### Table 4: Joint Distribution: Severity Grade × Acquisition Domain
| Severity Grade | `fruits360` | `real_grade` | `real_quality` | `real_severity` | Total |
|:---:|---:|---:|---:|---:|---:|
| **0 (Healthy)** | 43,962 | 837 | 3,478 | 779 | **49,056** |
| **1 (Mild)** | 0 | 892 | 0 | 536 | **1,428** |
| **2 (Moderate)** | 0 | 0 | 523 | 0 | **523** |
| **3 (Severe)** | 0 | 355 | 3,387 | 0 | **3,742** |
| **4 (Rotten)** | 0 | 0 | 0 | 1,287 | **1,287** |
| **Total** | **43,962** | **2,084** | **7,388** | **2,602** | **56,036** |

---

## 3. Investigation of the Missing Grade 2 Test Samples (Task 2.2)

### Diagnostic Trace:
We traced the entire label lifecycle from physical folder to metric reporting:
1. **Raw Folders:** `Apple_Mixed_Quality` ($n=113$), `Banana_Mixed_Quality` ($n=285$), and `Orange_Mixed Quality` ($n=125$). Total raw images = $523$.
2. **Severity Mapping Configuration (`severity_mapping.json`):**
   ```json
   "real_quality": { "good": 0, "mixed": 2, "bad": 3 }
   ```
   The mapping correctly translates `mixed` to integer ordinal index `2`.
3. **Session Assignment in `generate_master_csv.py`:**
   Filenames were parsed into session IDs by extracting date substrings `(\d{8})`.
   The 523 Grade 2 images belong to exactly three session dates:
   - `20200728`: Apple ($n=113$), Orange ($n=95$)
   - `20200729`: Banana ($n=166$), Orange ($n=30$)
   - `20200730`: Banana ($n=119$)
4. **Split Generation Defect:**
   Lines 138–166 of `generate_master_csv.py` executed:
   ```python
   for key, session_ids in fruit_domain_sessions.items():
       sid_list = sorted(session_ids)
       rng.shuffle(sid_list)  # Seed = 42
       # n_train = int(n * 0.70), n_val = int(n * 0.15)
   ```
   The grouping was performed at the `(fruit, domain)` level, completely blind to target class labels.
   Under `RANDOM_SEED = 42`:
   - For `(Apple, real_quality)` ($n=12$ sessions): `20200728` landed in `train`.
   - For `(Banana, real_quality)` ($n=22$ sessions): `20200729` landed in `val`, `20200730` landed in `train`.
   - For `(Orange, real_quality)` ($n=14$ sessions): `20200728` landed in `train`, `20200729` landed in `val`.
5. **Resulting Allocation:**
   - Train Grade 2: $113 + 119 + 95 = \mathbf{327}$
   - Validation Grade 2: $166 + 30 = \mathbf{196}$
   - Test Grade 2: $\mathbf{0}$

### Root Cause Identification:
- Genuine dataset absence: **NO** (523 real images exist).
- Incorrect severity mapping: **NO** (Mapped correctly to 2).
- Dropped or overwritten class in memory: **NO**.
- Split generation behavior: **YES — DEFECT CONFIRMED**.
- Filtering bug: **NO**.
- Metric-reporting bug: **NO**.

### Mathematical Impact on Evaluation:
Because test support for Grade 2 is $N_2 = 0$:
1. Any standard Macro-F1 implementation averaging over 5 classes assigns $F_{1, 2} = 0.0000$ (or undefined). This mathematically suppresses reported 5-class Macro-F1 by $20.0\%$.
2. The reported 5-class Macro-F1 of $\approx 29.83\%$ (Baseline) and $30.32\%$ (CASM v2) is artificially depressed. On the 4 active test classes, effective Macro-F1 is $\approx 37.9\%$.
3. The test set does not test whether the model can predict Grade 2. The ordinal ranking evaluation is strictly a 4-class evaluation on $\{0, 1, 3, 4\}$.

---

## 4. Ordinal Label Mapping Audit (Task 2.3)

### Systematic Domain Specification:
| Source Domain | Raw Folder Label | Mapped Name | Numerical Grade | Semantic Justification |
|---|---|---|:---:|---|
| `fruits360` | `all_varieties` | Healthy | **0** | Laboratory-isolated, unblemished agricultural specimens. |
| `real_grade` | `1st_grade` | Healthy | **0** | Prime commercial export quality; pristine fruit surface. |
| `real_grade` | `2nd_grade` | Mild Defect | **1** | Minor commercial blemishes; retail acceptable. |
| `real_grade` | `3rd_grade` | Severe Defect | **3** | Substandard processing grade; major visible flaws. |
| `real_quality` | `Good_Quality` | Healthy | **0** | Market inspection grade: pristine quality. |
| `real_quality` | `Mixed_Quality` | Moderate Defect | **2** | Intermediate commercial tier; heterogeneous quality. |
| `real_quality` | `Bad_Quality` | Severe Defect | **3** | Unmarketable / degraded condition. |
| `real_severity` | `Good` | Healthy | **0** | Biologically uninfected specimen. |
| `real_severity` | `Mild` | Mild Defect | **1** | Early stage infection, localized spots. |
| `real_severity` | `Rotten` | Rotten | **4** | Advanced necrotic fungal decay, total breakdown. |

### Semantic Monotonicity Verification:
$$\text{Healthy (0)} \prec \text{Mild Defect (1)} \prec \text{Moderate Defect (2)} \prec \text{Severe Defect (3)} \prec \text{Rotten (4)}$$
Monotonicity holds in physical degradation and commercial utility.
**Identified Semantic Ambiguity:**
- `real_grade` skips Grade 2 (maps 2nd_grade to 1 and 3rd_grade to 3).
- `real_quality` maps `mixed` to 2 and `bad` to 3, but skips 1 and 4.
- `real_severity` maps `mild` to 1 and `rotten` to 4, but skips 2 and 3.
This multi-source mosaic is semantically valid, but domain boundaries coincide exactly with grade boundaries.

---

## 5. Proposed Correction Plan for Versioned Split

### Requirements:
1. **Never Overwrite the Historical Split:** The historical split (`master_dataset.csv`, `split` column) must be preserved intact to guarantee 100% computational reproducibility of Phase 4.0–4.4 results.
2. **Construct Versioned Split (`split_v2`):**
   - Must preserve session-level grouping to prevent specimen/date leakage.
   - Must stratify session assignment by joint `(fruit, domain, severity_grade)` so that all 5 grades are represented in train, validation, AND test partitions.
   - For `real_quality` sessions containing `mixed` (Grade 2):
     - `20200728` (113 Apple, 95 Orange): Assign to `train` ($n=208$).
     - `20200729` (166 Banana, 30 Orange): Assign to `val` ($n=196$).
     - `20200730` (119 Banana): Reallocate session `20200730` from `train` to `test` ($n=119$).
     - This guarantees: Train Grade 2 = 208, Val Grade 2 = 196, Test Grade 2 = 119 ($N_{\text{test}, 2} > 0$).
   - This reallocation maintains zero session leakage, preserves all specimen boundaries, and enables true 5-grade ordinal evaluation.

---

## 6. Stage 2 Sign-Off Verdict

- **Inspected:** 56,038 dataset rows, 4 domains, 5 species, 12,076 real-world image hashes.
- **Verified:** Zero cross-split path leakage; zero cross-split content duplicates; exact cause of missing Grade 2 in test split; exact domain-grade confounding structure.
- **Failed / Anomalies:** Grade 2 has 0 test samples in historical split due to unstratified session allocation; `fruits360` has 0 validation samples and comprises 86.3% of the test set.
- **Safety to Proceed:** It is **SAFE** to proceed to Phase 3 (Model Implementation Audit).
