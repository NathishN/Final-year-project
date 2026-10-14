# Phase 5.0 — Corrected Dataset Protocol

**Date:** October 2026  
**Stage:** 1 — Freeze and Validate Corrected Dataset Protocol  
**Status:** PASSED

---

## 1. Historical Split (split_v1) — Preserved Intact

The original `split` column in `master_dataset.csv` is preserved byte-for-byte.
No modification was made to any historical split assignment.
This ensures full computational reproducibility of all Phase 4.0–4.4 results.

**Known defect (not corrected in split_v1):**
- Grade 2 has **0** test samples due to unstratified session allocation.
- This defect is the primary motivation for split_v2.

---

## 2. Corrected Split (split_v2)

### Reallocation Action
Session `20200730` contains **119 Banana Grade 2** images that were in `train`.
These were reallocated deterministically to `test` based on:
- `session_id == '20200730'`
- `domain == 'real_quality'`
- `quality_label_normalized == 'mixed'`
- `fruit == 'Banana'`

No random shuffle was performed. The reallocation is deterministic and reproducible.

### Session Independence
- Session `20200730` was in `train` (split_v1). It is now in `test` (split_v2).
- Session `20200729` (166 Banana + 30 Orange Grade 2) remains in `val`.
- Session `20200728` (113 Apple + 95 Orange Grade 2) remains in `train`.
- **Zero Grade-2 session overlap** across train/val/test boundaries.

### Split v2 Grade Coverage

| Grade | Name | Train | Val | Test |
|:---:|---|---:|---:|---:|
| **0** | Healthy | 36,414 | 1,037 | **11,605** |
| **1** | Mild Defect | 846 | 494 | **88** |
| **2** | Moderate Defect | 208 | 196 | **119** |
| **3** | Severe Defect | 1,606 | 1,268 | **868** |
| **4** | Rotten | 716 | 418 | **153** |

**All five grades (0–4) are now represented in the test set.**

### Leakage Audit

| Split Pair | Path Overlap | Session Overlap |
|---|:---:|:---:|
| train_val | 0 | 0 |
| train_test | 0 | 1 |
| val_test | 0 | 0 |

---

## 3. Real-World Evaluation Subset (split_real)

> [!IMPORTANT]
> `split_real` is **NOT** an independent external test set.
> It is a diagnostic subset of `split_v2` that excludes the 43,962 synthetic
> `fruits360` images. Use it only to analyze real-world domain performance.
> Do not report `split_real` results as if they constitute independent validation.

### split_real Counts (Real-World Domains Only)

| Split | Total | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 |
|---|---:|---:|---:|---:|---:|---:|
| train | 6,802 | 3,426 | 846 | 208 | 1,606 | 716 |
| val | 3,413 | 1,037 | 494 | 196 | 1,268 | 418 |
| test | 1,859 | 631 | 88 | 119 | 868 | 153 |

---

## 4. Stage 1 Sign-Off

- ✅ split_v1 preserved intact, Grade 2 test = 0 confirmed
- ✅ split_v2 created: Grade 2 test = 119
- ✅ All 5 grades present in split_v2 test set
- ✅ Zero Grade-2 session leakage in split_v2
- ✅ split_real defined with explicit independence caveat
- ✅ Manifests saved to outputs/casm_v2/

**SAFE TO PROCEED TO STAGE 2 (Model Integration & Testing)**