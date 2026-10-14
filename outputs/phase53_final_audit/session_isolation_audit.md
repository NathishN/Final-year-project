# Forensic Verification Report: Session-Held-Out Split Integrity

**Audit Timestamp:** 2026-10-14 10:13:30  
**Target Manifest:** `outputs/phase52_final_validation/session_holdout_manifest.csv`  
**Baseline Manifest:** `outputs/casm_v2/phase50_split_v2_manifest.csv`  

---

## 1. Summary of Verification Checks

| # | Verification Check | Status | Evidence / Findings |
|---|---|:---:|---|
| 1 | Unique session enumeration | **PASS** | Exactly 80 unique sessions cataloged. |
| 2 | Partition exclusivity | **PASS** | Train: 49, Val: 10, Test: 21 sessions. Intersections: 0. |
| 3 | Null / malformed session check | **PASS** | 0 nulls, 0 empty strings, 0 malformed identifiers. |
| 4 | Duplicate path / image check | **PASS** | Exactly 0 cross-split path duplicates across all 56,036 samples. |
| 5 | Session ID provenance | **PASS** | Parsed directly from photo date stamps (YYYYMMDD) via regex; zero label inference. |
| 6 | 120-run manifest adherence | **PASS** | All 120 retrained models verified in `outputs/phase52_final_validation/session_holdout_retrain/`. |
| 7 | Zero cross-split leakage | **PASS** | Assertions enforced in retraining code; validation used strictly for model selection. |
| 8 | Class & domain distributions | **PASS** | Recalculated for all partitions; all 5 grades present in train, val, and test. |
| 9 | 12,980 test sample explanation | **PASS** | Recomputed exactly: 12,833 + 147 non-G2 samples of session 20200730 = 12,980. |
| 10 | Feature cache leakage check | **PASS** | Frozen backbone, per-image standardization, LayerNorm (zero cross-sample leakage). |

## 2. Partition Session Allocation & Exclusivity Matrix

- **Total Unique Sessions:** 80
- **Train Sessions:** 49
- **Validation Sessions:** 10
- **Test Sessions:** 21

### Cross-Split Overlap Checks:
- $\text{Train} \cap \text{Val}$: 0 sessions
- $\text{Train} \cap \text{Test}$: 0 sessions
- $\text{Val} \cap \text{Test}$: 0 sessions

> **Verdict on Exclusivity:** `PASS`. Every one of the 80 acquisition sessions belongs strictly to a single partition.

## 3. Reconciliation: Historical Split V2 vs. Group Session Holdout

In historical `split_v2`, session `20200730` spanned train and test:
- **Test:** 119 Banana Grade 2 samples
- **Train:** 147 non-Grade-2 samples (15 Apple Grade 0, 85 Banana Grade 0, 47 Banana Grade 3)

In `session_holdout_manifest.csv`, all 266 samples of session `20200730` were assigned to `test`.

### Partition Sample Counts Comparison:

| Split | Historical Split V2 Samples | Group Session Holdout Samples | Difference |
|---|---:|---:|---:|
| Train | 39,790 | 39,643 | -147 |
| Val | 3,413 | 3,413 | 0 |
| Test | 12,833 | 12,980 | +147 |
| **Total** | **56,036** | **56,036** | **0** |

### Target Severity Grade Distribution in Group Session Holdout:

| Partition | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Total |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 36,314 | 846 | 208 | 1,559 | 716 | **39,643** |
| `val` | 1,037 | 494 | 196 | 1,268 | 418 | **3,413** |
| `test` | 11,705 | 88 | 119 | 915 | 153 | **12,980** |

## 4. Scientific Verdict

The claim that `session_holdout_manifest.csv` provides **100% group isolation across all 80 acquisition sessions** is **SUBSTANTIATED AND FULLY VERIFIED**.
