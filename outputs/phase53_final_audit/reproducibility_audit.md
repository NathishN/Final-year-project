# Forensic Results-to-Code Matrix & Reproducibility Audit

**Audit Timestamp:** 2026-10-14 10:25:20  

**Repository Test Suite Status:** 17/17 tests passing (`Ran 17 tests in 5.511s, OK`)  

**Checkpoints Status:** 120 historical + 120 retrained = 240 model files fully verified  

---

## 1. Headline Results-to-Code Traceability Matrix

| Claim ID | Scientific Claim | Source Code | Checkpoints / Artifacts | Reproduction Command | Verification Status |
|---|---|---|---|---|:---:|
| **CLM-001** | MonotonicCORAL guarantees strictly ordered cutpoint thresholds and valid ordinal probabilities | `src/models/coral_head.py` | `outputs/phase53_final_audit/test_results.txt` | `python -m unittest tests/test_model_properties.py -v` | **VERIFIED** |
| **CLM-002** | Config A baseline achieves 92.11% +- 0.94% accuracy, QWK=0.8712 +- 0.0249, MAE=0.1145 +- 0.0166 across 30 seeds | `phase50_stage5_confirmatory.py` | `outputs/phase53_final_audit/statistical_reproduction.csv` | `python outputs/phase53_final_audit/audit_section3_statistics.py` | **VERIFIED** |
| **CLM-003** | Exemplar single run (Seed 83) achieves 93.71% accuracy, 0.0889 MAE, 1.45% ECE | `phase50_stage5_confirmatory.py` | `outputs/phase53_final_audit/calibration_provenance.json` | `python outputs/phase53_final_audit/audit_section4_calibration.py` | **VERIFIED** |
| **CLM-004** | Species conditioning degrades ordinal grading (B vs A: mean difference -0.0360 QWK, t=-4.159, p_adj=0.000260) | `phase50_stage5_confirmatory.py` | `outputs/phase53_final_audit/statistical_reproduction.md` | `python outputs/phase53_final_audit/audit_section3_statistics.py` | **VERIFIED (NEGATIVE RESULT)** |
| **CLM-005** | Negative result confirmed on 100% group session-held-out split (B vs A: mean difference -0.0262 QWK, p_adj=0.01018) | `outputs/phase52_final_validation/run_session_holdout_retrain.py` | `outputs/phase53_final_audit/session_isolation_audit.md` | `python outputs/phase53_final_audit/audit_section2_session_isolation.py` | **VERIFIED (NEGATIVE RESULT)** |
| **CLM-006** | Energy score detects unseen fruit species (81.99% overall AUROC, 92.49% on natural market images) | `phase50_stage4_real_ood.py` | `outputs/phase53_final_audit/ood_domain_reconciliation.md` | `python outputs/phase53_final_audit/audit_section5_ood.py` | **VERIFIED** |
| **CLM-007** | Mahalanobis detector collapses on natural market photography (AUROC 18.08% on real_quality vs 93.26% on studio) | `phase50_stage4_real_ood.py` | `outputs/phase53_final_audit/ood_domain_reconciliation.md` | `python outputs/phase53_final_audit/audit_section5_ood.py` | **VERIFIED (FAILURE MODE)** |
| **CLM-008** | Low pooled ECE (1.45% - 2.51%) is studio-dominated (85.5% fruits360); market bin calibration degrades to 10.29% - 18.42% | `outputs/phase53_final_audit/audit_section4_calibration.py` | `outputs/phase53_final_audit/calibration_provenance.md` | `python outputs/phase53_final_audit/audit_section4_calibration.py` | **VERIFIED (LIMITATION)** |
| **CLM-009** | Selective prediction discarding least-confident 10% samples reduces MAE from 0.0889 to 0.0136 (84.7% error drop) | `outputs/phase53_final_audit/audit_section4_calibration.py` | `outputs/phase53_final_audit/calibration_provenance.md` | `python outputs/phase53_final_audit/audit_section4_calibration.py` | **VERIFIED** |
| **CLM-010** | Full end-to-end pipeline achieves 31.2 FPS (BS=1) and 75.6 FPS (BS=32) in FP16 on RTX 3050; Head-only is >100,000 FPS | `outputs/phase53_final_audit/audit_section7_benchmarks.py` | `outputs/phase53_final_audit/e2e_benchmark_reproduction.md` | `python outputs/phase53_final_audit/audit_section7_benchmarks.py` | **VERIFIED** |


## 2. Test Suite Execution Provenance

The repository test suite was executed from clean state via `python -m unittest discover tests -v`:

```text
test_all_five_grades_present (test_data_integrity.TestDataIntegrity.test_all_five_grades_present)
All five severity grades (0, 1, 2, 3, 4) must exist in manifest. ... ok
test_grade2_test_coverage (test_data_integrity.TestDataIntegrity.test_grade2_test_coverage)
Test set must contain exactly 119 Grade 2 samples from session 20200730. ... ok
test_total_valid_samples (test_data_integrity.TestDataIntegrity.test_total_valid_samples)
Manifest must contain exactly 56,036 active valid samples. ... ok
test_zero_cross_split_path_leakage (test_data_integrity.TestDataIntegrity.test_zero_cross_split_path_leakage)
Zero image paths may cross train, val, or test splits. ... ok
test_zero_grade2_session_leakage (test_data_integrity.TestDataIntegrity.test_zero_grade2_session_leakage)
Session 20200730 in Banana Grade 2 must exist exclusively in test split. ... ok
test_forward_backward_pass (test_end_to_end_smoke.TestEndToEndSmoke.test_forward_backward_pass)
Forward pass, loss computation, and backprop must complete with finite gradients. ... ok
test_all_120_prediction_files_present (test_integration_predictions.TestIntegrationPredictions.test_all_120_prediction_files_present)
All 120 confirmatory runs (30 seeds x 4 configs) must exist on disk. ... ok
test_confirmatory_statistical_reconciliation (test_integration_predictions.TestIntegrationPredictions.test_confirmatory_statistical_reconciliation)
Config A vs Config B 30-seed contrast matches forensic statistical values. ... ok
test_prediction_tensor_structure_and_probabilities (test_integration_predictions.TestIntegrationPredictions.test_prediction_tensor_structure_and_probabilities)
Spot-check prediction tensors for shape, valid probability simplex, and finite values. ... ok
test_recomputed_metrics_match_stored (test_integration_predictions.TestIntegrationPredictions.test_recomputed_metrics_match_stored)
Recomputed QWK and MAE from tensors must match stored checkpoint metadata within rounding. ... ok
test_degenerate_qwk_detection (test_metrics_and_calibration.TestMetricsAndCalibration.test_degenerate_qwk_detection)
Constant true labels produce zero label variance where QWK is mathematically uninformative. ... ok
test_ece_perfect_calibration (test_metrics_and_calibration.TestMetricsAndCalibration.test_ece_perfect_calibration)
Expected Calibration Error must be near 0 on perfectly calibrated probabilities. ... ok
test_qwk_perfect_and_worst (test_metrics_and_calibration.TestMetricsAndCalibration.test_qwk_perfect_and_worst)
QWK must equal 1.0 on perfect agreement and <= 0.0 on inverse agreement. ... ok
test_class_probs_valid_distribution (test_model_properties.TestModelProperties.test_class_probs_valid_distribution)
Discrete class probabilities must be non-negative and sum to 1.0 within 1e-5. ... ok
test_cutpoint_strictly_decreasing (test_model_properties.TestModelProperties.test_cutpoint_strictly_decreasing)
MonotonicCORAL biases must satisfy b_0 > b_1 > b_2 > b_3 across 20 random seeds. ... ok
test_extreme_logit_stability (test_model_properties.TestModelProperties.test_extreme_logit_stability)
Extreme input values (+/-1000) must produce finite outputs without NaN or Inf. ... ok
test_unknown_species_routing (test_model_properties.TestModelProperties.test_unknown_species_routing)
Negative and out-of-range IDs must map safely to index 5 without exception. ... ok

----------------------------------------------------------------------
Ran 17 tests in 5.511s

OK
```

## 3. Independent Audit Findings on Reproducibility

1. **Complete Deterministic Reproducibility:** All metric claims in the paper trace directly to persisted prediction tensors (`preds_*.pt`) and checkpoints (`model_*.pth`).

2. **Hardware Environment:** All benchmarks reproduce on standard commodity hardware (NVIDIA RTX 3050 Laptop GPU / Python 3.12 / PyTorch 2.5.1+cu121).

3. **Scientific Integrity of Claims:** There are zero unbacked claims. All previously ambiguous assertions (e.g. single-run vs aggregate means, head-only vs E2E throughput, background confounding vs causal claim) have been disambiguated and substantiated with code and data.

