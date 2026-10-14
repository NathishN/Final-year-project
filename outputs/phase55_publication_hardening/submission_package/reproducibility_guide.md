# Phase 5.5 — End-to-End Reproducibility Guide & Certification Protocol

**Guide Certified:** 2026-10-14 11:00:41  

**Auditing Role:** Reproducibility Reviewer & Open Science Auditor  

**Repository Root:** `C:\fruit_dataset`  

---

## 1. Reproduction Philosophy & Scope

This repository certifies **exact deterministic numerical reproducibility** from saved feature tensors and prediction files:

- **Reproducing Reported Metrics:** Every number, table cell, statistical $p$-value, calibration metric, and figure in the paper can be deterministically reproduced in under 60 seconds without retraining.

- **Full Retraining from Scratch:** Requires 120 independent training runs per protocol (15 epochs, AdamW optimizer, batch size 2,048) on precomputed 768-D ConvNeXt-Tiny embeddings. Total compute time is ~2.5 hours on an NVIDIA RTX 3050 Laptop GPU.


## 2. Master Reproduction Matrix

| Step | Workflow Stage | Execution Command | Inputs | Outputs | Verification Criterion |
|:---:|---|---|---|---|---|
| **1** | Environment & Package Audit | `python -c "import torch, torchvision, sklearn, scipy, pandas; print('PyTorch:', torch.__version__)"` | `Python 3.12, CUDA Toolkit` | `Console output` | PyTorch >= 2.5 with CUDA or CPU support |
| **2** | Dataset & Manifest Integrity | `python outputs/phase55_publication_hardening/certify_split_integrity.py` | `outputs/casm_v2/phase50_split_v2_manifest.csv, outputs/phase52_final_validation/session_holdout_manifest.csv` | `outputs/phase55_publication_hardening/split_integrity.json` | 56,036 samples, 0 duplicate paths, 0 cross-split sessions in holdout |
| **3** | Checkpoint & Prediction Tensor Census | `python outputs/phase55_publication_hardening/audit_inventory.py` | `outputs/casm_v2/phase50_checkpoints/, outputs/phase52_final_validation/session_holdout_retrain/checkpoints/` | `outputs/phase55_publication_hardening/INITIAL_INVENTORY.md` | 120 historical + 120 retrained = 240 model checkpoints verified |
| **4** | Confirmatory Statistical Contrasts | `python outputs/phase55_publication_hardening/recompute_statistics_final.py` | `outputs/casm_v2/phase50_per_seed_predictions/, outputs/phase52_final_validation/session_holdout_retrain/per_seed_results.csv` | `outputs/phase55_publication_hardening/statistical_final.json, statistical_tables.csv` | Historical B-A p_adj=0.000260; Session holdout B-A p_adj=0.01018 |
| **5** | Selective Prediction & Calibration | `python outputs/phase55_publication_hardening/reconcile_selective_metrics.py` | `outputs/casm_v2/phase50_per_seed_predictions/preds_A_seed83.pt` | `outputs/phase55_publication_hardening/selective_metric_reconciliation.json` | Baseline MAE 0.0889 -> Selective MAE 0.0136 (84.7% reduction) |
| **6** | Real-Image OOD Recomputation | `python outputs/phase54_discrepancy_resolution/resolve_issue4_ood_metrics.py` | `outputs/casm_v2/phase50_ood_features/ood_real_features.pt, outputs/features/phase4_dataset.pt` | `outputs/phase54_discrepancy_resolution/issue4_ood_metrics_recomputation.json` | Mahalanobis AUROC 84.22% pooled, 18.08% market; Energy 81.99% pooled, 92.49% market |
| **7** | Publication Figures Generation | `python outputs/phase55_publication_hardening/generate_publication_figures.py` | `All audited JSON data files in outputs/phase55_publication_hardening/` | `outputs/phase55_publication_hardening/figures/*.png (6 figures at 300 DPI)` | All 6 high-resolution publication figures generated |
| **8** | Full Regression & Unit Test Suite | `python -m unittest discover tests -v` | `tests/test_*.py` | `outputs/phase55_publication_hardening/test_results.txt` | Ran 17 tests with exit code 0 (OK) |
| **9** | Manuscript Package Assembly | `python scripts/assemble_paper.py` | `paper/manuscript.md, paper/references.bib, paper/limitations.md` | `paper/ distribution bundle` | Exit code 0, complete package built |


## 3. Current Test Suite Certification Log

The 17-test regression and property suite was executed from a clean state:

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
Ran 17 tests in 4.383s

OK

```


## 4. Reproducibility Certification Verdict

```text

========================================================================================

                       REPRODUCIBILITY CERTIFIED FOR SUBMISSION

           (All empirical claims, figures, tables, and tests 100% verifiable)

========================================================================================

```

