# Phase 5.3 — Initial Repository and Artifact Inventory

**Date of Audit:** 2026-10-14 09:20:50  
**Auditor:** Senior Forensic Reviewer, Computer Vision & Statistics  
**Repository Root:** `C:\fruit_dataset`  

---

## 1. Executive Summary of Available Evidence

- **Key Tracked Artifacts:** 55 / 55 present on disk.
- **Historical Confirmatory Checkpoints:** 120 / 120 present in `outputs/casm_v2/phase50_checkpoints/`.
- **Historical Confirmatory Predictions:** 120 / 120 present in `outputs/casm_v2/phase50_per_seed_predictions/`.
- **Session-Held-Out Retrained Checkpoints:** 120 / 120 present in `outputs/phase52_final_validation/session_holdout_retrain/checkpoints/`.
- **Session-Held-Out Retrained Predictions:** 120 / 120 present in `outputs/phase52_final_validation/session_holdout_retrain/predictions/`.
- **Master Dataset Feature Cache:** `outputs/features/phase4_dataset.pt` (179.2 MB, 56,036 samples, 768-D).
- **Real OOD Feature Cache:** `outputs/casm_v2/phase50_ood_features/ood_real_features.pt` (15.9 MB, 5,000 samples across 5 unseen species).

## 2. Inventory of Critical Files

| Relative Path | Exists | Size (Bytes) | Last Modified |
|---|:---:|---:|:---|
| `outputs/phase52_final_validation/FINAL_REPORT.md` | YES | 7,905 | 2026-10-14T09:12:54.939899 |
| `outputs/phase52_final_validation/CLAIM_LEDGER.csv` | YES | 3,047 | 2026-10-13T18:18:06.614219 |
| `outputs/phase52_final_validation/PROGRESS.md` | YES | 3,391 | 2026-10-14T09:13:09.270233 |
| `outputs/phase52_final_validation/COMMAND_LOG.md` | YES | 1,223 | 2026-10-13T18:18:06.619780 |
| `outputs/phase52_final_validation/DECISIONS.md` | YES | 1,052 | 2026-10-13T18:18:06.621157 |
| `outputs/phase52_final_validation/ISSUES.csv` | YES | 49 | 2026-10-13T18:18:06.619780 |
| `outputs/phase52_final_validation/run_verification_matrix.csv` | YES | 16,090 | 2026-10-13T18:13:10.107128 |
| `outputs/phase52_final_validation/run_verification_report.md` | YES | 1,031 | 2026-10-13T18:13:10.113343 |
| `outputs/phase52_final_validation/training_protocol_verification.md` | YES | 1,025 | 2026-10-13T18:13:10.114349 |
| `outputs/phase52_final_validation/recomputed_metrics_by_configuration.csv` | YES | 710 | 2026-10-13T18:13:11.246636 |
| `outputs/phase52_final_validation/recomputed_metrics_by_seed.csv` | YES | 7,542 | 2026-10-13T18:13:11.243367 |
| `outputs/phase52_final_validation/metric_discrepancies.csv` | YES | 237 | 2026-10-13T18:13:11.249635 |
| `outputs/phase52_final_validation/paired_differences.csv` | YES | 2,150 | 2026-10-13T18:13:11.260544 |
| `outputs/phase52_final_validation/statistical_reconciliation.md` | YES | 2,084 | 2026-10-13T18:13:11.279531 |
| `outputs/phase52_final_validation/split_audit.csv` | YES | 4,879 | 2026-10-13T18:13:11.603741 |
| `outputs/phase52_final_validation/split_audit_report.md` | YES | 689 | 2026-10-13T18:13:11.605946 |
| `outputs/phase52_final_validation/session_holdout_manifest.csv` | YES | 6,212,673 | 2026-10-13T18:13:11.988767 |
| `outputs/phase52_final_validation/session_holdout_results.csv` | YES | 1,125 | 2026-10-14T09:12:13.712772 |
| `outputs/phase52_final_validation/session_holdout_report.md` | YES | 3,275 | 2026-10-14T09:12:28.469313 |
| `outputs/phase52_final_validation/ood_metrics_recomputed.csv` | YES | 168 | 2026-10-13T18:13:14.901953 |
| `outputs/phase52_final_validation/ood_species_results.csv` | YES | 347 | 2026-10-13T18:13:14.904959 |
| `outputs/phase52_final_validation/ood_domain_results.csv` | YES | 205 | 2026-10-13T18:13:14.906958 |
| `outputs/phase52_final_validation/ood_operating_points.csv` | YES | 153 | 2026-10-13T18:13:14.908570 |
| `outputs/phase52_final_validation/ood_confounding_report.md` | YES | 2,062 | 2026-10-13T18:13:14.909582 |
| `outputs/phase52_final_validation/calibration_recomputed.csv` | YES | 354 | 2026-10-13T18:13:14.919703 |
| `outputs/phase52_final_validation/selective_prediction_results.csv` | YES | 479 | 2026-10-13T18:13:14.925706 |
| `outputs/phase52_final_validation/calibration_and_abstention_report.md` | YES | 1,363 | 2026-10-13T18:13:14.926706 |
| `outputs/phase52_final_validation/e2e_benchmark_results.csv` | YES | 1,345 | 2026-10-13T18:16:36.540634 |
| `outputs/phase52_final_validation/e2e_benchmark_report.md` | YES | 2,045 | 2026-10-13T18:16:36.544084 |
| `outputs/phase52_final_validation/reproduction_command_results.csv` | YES | 1,108 | 2026-10-14T08:15:02.910011 |
| `outputs/phase52_final_validation/integration_test_audit.md` | YES | 1,517 | 2026-10-14T08:15:04.965414 |
| `outputs/phase52_final_validation/session_holdout_retrain/summary_by_config.csv` | YES | 485 | 2026-10-14T08:29:57.893554 |
| `outputs/phase52_final_validation/session_holdout_retrain/paired_stats.csv` | YES | 682 | 2026-10-14T08:29:57.922086 |
| `outputs/phase52_final_validation/session_holdout_retrain/per_seed_results.csv` | YES | 14,538 | 2026-10-14T08:29:57.854992 |
| `outputs/checkpoints/best_model.pth` | YES | 334,116,182 | 2026-10-07T00:50:53.217863 |
| `outputs/features/phase4_dataset.pt` | YES | 179,232,590 | 2026-10-08T10:27:53.132512 |
| `outputs/casm_v2/phase50_split_v2_manifest.csv` | YES | 6,212,820 | 2026-10-13T15:33:16.856078 |
| `outputs/casm_v2/phase50_ood_features/ood_real_features.pt` | YES | 15,875,604 | 2026-10-13T15:52:21.421639 |
| `outputs/casm_v2/phase50_ood_features/features_metadata.json` | YES | 2,838 | 2026-10-13T15:55:32.861248 |
| `outputs/final_audit/ood_predictions.csv` | YES | 583,471 | 2026-10-13T16:21:17.390615 |
| `paper/manuscript.md` | YES | 15,917 | 2026-10-14T09:13:32.627645 |
| `paper/abstract.md` | YES | 2,962 | 2026-10-14T09:13:32.619132 |
| `paper/methodology.md` | YES | 5,287 | 2026-10-14T09:13:32.622133 |
| `paper/results.md` | YES | 5,035 | 2026-10-14T09:13:32.624133 |
| `paper/limitations.md` | YES | 1,252 | 2026-10-14T09:13:32.625132 |
| `paper/references.bib` | YES | 2,540 | 2026-10-14T09:13:32.620133 |
| `paper/final_reproducibility_checklist.md` | YES | 3,760 | 2026-10-13T19:17:21.099029 |
| `paper/final_claim_evidence_matrix.csv` | YES | 3,047 | 2026-10-13T18:18:06.616199 |
| `paper/publication_readiness_assessment.md` | YES | 3,237 | 2026-10-13T19:17:48.962854 |
| `paper/manuscript_revision_log.md` | YES | 2,460 | 2026-10-13T19:18:22.505027 |
| `tests/test_data_integrity.py` | YES | 2,269 | 2026-10-13T16:24:35.340166 |
| `tests/test_model_properties.py` | YES | 2,648 | 2026-10-13T16:24:46.373196 |
| `tests/test_metrics_and_calibration.py` | YES | 1,665 | 2026-10-13T16:24:56.160143 |
| `tests/test_end_to_end_smoke.py` | YES | 1,918 | 2026-10-13T16:25:09.447763 |
| `tests/test_integration_predictions.py` | YES | 5,359 | 2026-10-13T17:05:51.526621 |

## 3. Scripts and Corresponding Regeneration Commands

| Result Domain | Source Script | Input Artifacts | Output Artifacts | Exact Command |
|---|---|---|---|---|
| Unit & Integration Tests | `tests/` | Prediction tensors, models | Test execution log | `python -m unittest discover tests -v` |
| Split & Dataset Integrity | `scripts/phaseB_dataset_integrity.py` | `outputs/casm_v2/phase50_split_v2_manifest.csv` | `outputs/final_audit/split_integrity.csv` | `python scripts/phaseB_dataset_integrity.py` |
| Ordinal Loss & Cutpoints | `scripts/phaseC_ordinal_loss_audit.py` | `phase50_casm_v2_model.py` | Validation logs | `python scripts/phaseC_ordinal_loss_audit.py` |
| Hardware Latency & E2E | `scripts/phaseH_model_efficiency.py` | `outputs/checkpoints/best_model.pth` | `outputs/final_audit/latency_benchmarks.csv` | `python scripts/phaseH_model_efficiency.py` |
| Manuscript Assembly | `scripts/assemble_paper.py` | `paper/*.md`, `references.bib` | `paper/manuscript.md` | `python scripts/assemble_paper.py` |
| 120 Historical Runs | `phase50_stage5_confirmatory.py` | `outputs/features/phase4_dataset.pt` | `outputs/casm_v2/phase50_checkpoints/`, `phase50_per_seed_predictions/` | `python phase50_stage5_confirmatory.py` |
| 120 Session Holdout Runs | `outputs/phase52_final_validation/run_session_holdout_retrain.py` | `session_holdout_manifest.csv`, `phase4_dataset.pt` | `outputs/phase52_final_validation/session_holdout_retrain/` | `python outputs/phase52_final_validation/run_session_holdout_retrain.py` |

## 4. Inconsistencies & High-Risk Questions to Reconcile

1. **Session-Held-Out Protocol Integrity:** In `split_v2`, Session `20200730` had 119 Grade 2 Banana images in test, but 147 samples in train. Phase 5.2 generated `session_holdout_manifest.csv` (12,980 test samples) and retrained all 120 runs. Must independently audit every session ID, sample count, hash, and leakage vector.
2. **Statistical Reconciliation:** B vs A degradation reported as $-0.0360$ (historical) and $-0.0262$ (session-held-out). Wilcoxon $W=72.0$ vs prior citations of 59 and 62 must be verified mathematically with exact zero handling, tie handling, and Holm-Bonferroni correction.
3. **Calibration Provenance:** Recomputed ECE of 1.45% (10 bins) vs historical 2.51% and selective top-90% accuracy of 99.28% vs 99.14% must be traced to exact split, checkpoint, and validation-tuning rules. Domain-stratified calibration must expose studio dominance.
4. **OOD Domain-Mapping Inconsistency:** Clarify the exact correspondence between `real_quality`, `real_grade`, and conveyor/market descriptors in text and tables. Audit Mahalanobis inversion (18.08%) vs Energy score (92.49% on `real_quality`, 98.89% on `real_grade`). Confirm why causal background attribution is scientifically unresolved.
5. **Hardware Benchmark Verification:** Verify end-to-end inference benchmark on RTX 3050 (including image loading, preprocessing, backbone, and head) versus head-only throughput.
