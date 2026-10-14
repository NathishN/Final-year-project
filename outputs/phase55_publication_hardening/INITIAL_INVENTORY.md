# Phase 5.5 — Complete Project Inventory & Artifact Census

**Audit Timestamp:** 2026-10-14 10:53:37  

**Auditing Role:** Senior Computer Vision Researcher & Reproducibility Reviewer  

**Compute Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA Available: True), PyTorch 2.5.1+cu121  

---

## 1. Primary Artifact Verification Census

| Critical Artifact Path | Exists | Size (Bytes) | Last Modified | Verification Meaning |
|---|:---:|---:|:---|---|
| `outputs/phase53_final_audit/FINAL_AUDIT_REPORT.md` | **YES** | 13,532 | 2026-10-14 10:27:40 | Core Artifact |
| `outputs/phase54_discrepancy_resolution/FINAL_REPORT.md` | **YES** | 11,306 | 2026-10-14 10:45:18 | Phase 5.4 Report |
| `outputs/phase54_discrepancy_resolution/discrepancy_ledger.csv` | **YES** | 4,505 | 2026-10-14 10:45:18 | Phase 5.4 Report |
| `outputs/phase54_discrepancy_resolution/issue1_statistical_resolution.json` | **YES** | 3,973 | 2026-10-14 10:35:25 | Phase 5.4 Report |
| `outputs/phase54_discrepancy_resolution/issue2_session_isolation.json` | **YES** | 3,446 | 2026-10-14 10:36:05 | Phase 5.4 Report |
| `outputs/phase54_discrepancy_resolution/issue3_calibration_traceability.json` | **YES** | 4,517 | 2026-10-14 10:38:17 | Phase 5.4 Report |
| `outputs/phase54_discrepancy_resolution/issue4_ood_metrics_recomputation.json` | **YES** | 4,757 | 2026-10-14 10:39:33 | Phase 5.4 Report |
| `outputs/phase54_discrepancy_resolution/issue5_benchmarks.json` | **YES** | 5,547 | 2026-10-14 10:42:20 | Phase 5.4 Report |
| `outputs/casm_v2/phase50_split_v2_manifest.csv` | **YES** | 6,212,820 | 2026-10-13 15:33:16 | Core Manifest |
| `outputs/phase52_final_validation/session_holdout_manifest.csv` | **YES** | 6,212,673 | 2026-10-13 18:13:11 | Core Manifest |
| `outputs/features/phase4_dataset.pt` | **YES** | 179,232,590 | 2026-10-08 10:27:53 | Core Artifact |
| `outputs/casm_v2/phase50_ood_features/ood_real_features.pt` | **YES** | 15,875,604 | 2026-10-13 15:52:21 | Core Artifact |
| `outputs/casm_v2/phase50_ood_features/features_metadata.json` | **YES** | 2,838 | 2026-10-13 15:55:32 | Core Artifact |
| `outputs/checkpoints/best_model.pth` | **YES** | 334,116,182 | 2026-10-07 00:50:53 | Core Artifact |
| `outputs/final_audit/ood_predictions.csv` | **YES** | 583,471 | 2026-10-13 16:21:17 | Core Artifact |
| `configs/training_protocol.json` | **YES** | 1,580 | 2026-10-13 18:17:03 | Core Artifact |
| `paper/manuscript.md` | **YES** | 15,917 | 2026-10-14 10:44:12 | Manuscript File |
| `paper/limitations.md` | **YES** | 1,252 | 2026-10-14 10:44:12 | Manuscript File |
| `paper/references.bib` | **YES** | 2,540 | 2026-10-14 10:44:12 | Manuscript File |
| `scripts/assemble_paper.py` | **YES** | 20,984 | 2026-10-13 17:17:00 | Manuscript File |


## 2. Model Checkpoints & Prediction Tensors Census

| Artifact Category | Location Directory | Expected Count | Verified on Disk | Total Disk Size |
|---|---|---:|---:|---:|
| **Historical Confirmatory Checkpoints** | `outputs/casm_v2/phase50_checkpoints/` | 120 | **120** | 288.4 MB |
| **Historical Prediction Tensors** | `outputs/casm_v2/phase50_per_seed_predictions/` | 120 | **120** | 53.1 MB |
| **Session-Held-Out Retrained Checkpoints** | `outputs/phase52_final_validation/session_holdout_retrain/checkpoints/` | 120 | **120** | 288.4 MB |
| **Session-Held-Out Retrained Predictions** | `outputs/phase52_final_validation/session_holdout_retrain/predictions/` | 120 | **120** | 53.7 MB |
| **Regression & Integration Test Suite** | `tests/` | 4+ test modules | **5** modules | N/A (17 test cases) |


## 3. Discrepancy Status & Evidence Classifications

To maintain scientific integrity across Phase 5.5, all evidence is strictly partitioned into:

1. **Category A (Historical Confirmatory Evidence):** 120 models evaluated on `split_v2` (12,833 test samples; Banana G2 in test, Apple G0 from session 20200730 in train). Stored in `outputs/casm_v2/`.

2. **Category B (Clean Group-Held-Out Evidence):** 120 models retrained from scratch on `session_holdout_manifest.csv` (12,980 test samples; all 266 samples of session 20200730 in test). Stored in `outputs/phase52_final_validation/session_holdout_retrain/`.

3. **Category C (Independently Recomputed in Phase 5.4/5.5):** Statistical contrasts, calibration curves, OOD metrics, and hardware latencies recomputed directly from raw tensors without model alterations.

4. **Category D (Explicitly Unverifiable / Bounded Claims):** Background causal attribution (bounded as `UNRESOLVED` due to lack of pixel segmentation masks); physical conveyor sorting line readiness (bounded as `UNVALIDATED` on physical hardware).


## 4. Pending Numerical Inconsistencies for Phase 5.5 Resolution

1. **Selective MAE Reduction Arithmetic:** Phase 5.4 reported an MAE drop from 0.1234 to 0.0136 as '84.7% reduction', whereas $(0.1234 - 0.0136)/0.1234 = 88.98%$, and an 84.7% reduction corresponds to baseline MAE 0.0889 ($(0.0889 - 0.0136)/0.0889 = 84.70%)$. Phase 5.5 must reconcile the exact baseline (raw argmax MAE vs cutpoint-calibrated MAE).

2. **Calibration Decoupling:** Reconcile single-seed (Seed 83) vs 30-seed aggregate means for accuracy, MAE, ECE, Brier score, and coverage.

3. **Full Statistical Table Generation:** Generate machine-readable statistical tables for all contrasts ($B-A, C-A, D-A$) across both historical and session-held-out splits with Holm-Bonferroni corrections.

