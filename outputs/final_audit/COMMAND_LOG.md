# Phase 5.0 / Final Master Audit — Command Log

This log records every major audit, verification, and experiment execution command, with timestamps, exit codes, and deliverables.

| Timestamp | Step / Phase | Command Executed | Exit Code | Deliverables Produced |
|:---|:---|:---|:---:|:---|
| 2026-10-13 15:40 | Stage 1 (Splits) | `python phase50_stage1_dataset_splits.py` | 0 | `outputs/casm_v2/phase50_split_v2_manifest.csv`, `phase50_dataset_protocol.md` |
| 2026-10-13 15:44 | Stage 2 (Models) | `python phase50_stage2_model_integration.py` | 0 | `outputs/casm_v2/phase50_unit_test_results.txt`, `phase50_corrected_model_audit.md` |
| 2026-10-13 15:49 | Stage 3 (Pilot) | `python phase50_stage3_loss_pilot.py` | 0 | `outputs/casm_v2/phase50_loss_pilot_results.csv`, `phase50_loss_and_pilot_report.md` |
| 2026-10-13 15:55 | Stage 4 (Real OOD) | `python phase50_stage4_real_ood.py` | 0 | `outputs/casm_v2/phase50_ood_features/ood_real_features.pt`, `phase50_ood_benchmark.csv` |
| 2026-10-13 16:10 | Stage 5 (120 Runs) | `python phase50_stage5_confirmatory.py` | 0 | 120 Checkpoints, 120 Prediction files, `phase50_confirmatory_results.csv` |
| 2026-10-13 16:10 | Stage 6 & 7 (Domain/Calib) | `python phase50_stage6_7_eval.py` | 0 | `phase50_domain_generalization_report.md`, `phase50_calibration_report.md` |
| 2026-10-13 16:10 | Stage 8 (Master Report) | `python phase50_stage8_final_decision.py` | 0 | `outputs/casm_v2/phase50_final_research_decision.md` |
| 2026-10-13 16:15 | Master Audit Init | Directory & State Setup | 0 | `outputs/final_audit/` structure initialized |
