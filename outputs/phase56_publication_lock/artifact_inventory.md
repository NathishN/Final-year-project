# Phase 5.6 — Authoritative Publication Artifact Inventory

**Audit Timestamp:** 2026-10-14 13:42:37  

**Audit Scope:** Forensic cataloging of raw experimental checkpoints, predictions, manifests, OOD features, calibration outputs, benchmarks, and regression tests.  

**Principle:** Every final scientific claim must map directly to an entry in this inventory.

---

| # | Category | Authoritative Path / Pattern | Sample Count | Seed / Config Count | Verification Status | Purpose |
|:---:|:---|:---|:---|:---|:---|:---|
| 1 | **Historical Split V2 Checkpoints** | `outputs/casm_v2/phase50_checkpoints/model_[A-D]_seed*_best.pth` | 12,833 test samples per file | 120/120 files (30 seeds x 4 configs) | VERIFIED (120/120 present on disk, 626,180 params each) | Primary historical baseline weights across 30 seeds |
| 2 | **Historical Split V2 Predictions** | `outputs/casm_v2/phase50_per_seed_predictions/preds_[A-D]_seed*.pt` | 12,833 test predictions per file | 120/120 files (30 seeds x 4 configs) | VERIFIED (120/120 present, valid QWK/MAE/Accuracy/Probs) | Saved predictions for paired parametric & non-parametric evaluation |
| 3 | **Session-Held-Out Checkpoints** | `outputs/phase52_final_validation/session_holdout_retrain/checkpoints/model_[A-D]_seed*_best.pth` | 12,980 test samples per file | 120/120 files (30 seeds x 4 configs) | VERIFIED (120/120 present on disk, 100% session isolated) | Replication model weights with zero session leakage across partitions |
| 4 | **Session-Held-Out Predictions** | `outputs/phase52_final_validation/session_holdout_retrain/predictions/preds_[A-D]_seed*.pt` | 12,980 test predictions per file | 120/120 files (30 seeds x 4 configs) | VERIFIED (120/120 present, valid QWK/MAE/Accuracy/Probs) | Saved predictions for paired replication testing under strict holdout |
| 5 | **Session-Held-Out Results Master CSV** | `outputs/phase52_final_validation/session_holdout_retrain/per_seed_results.csv` | 120 run rows | 30 seeds x 4 configs | VERIFIED (contains test_qwk, test_mae, test_acc, test_macro_f1) | Tabular per-seed results for session holdout verification |
| 6 | **Dataset Manifest (Historical Split V2)** | `outputs/casm_v2/phase50_split_v2_manifest.csv` | 64,163 total rows (train=44,914, val=6,416, test=12,833) | N/A (split definition) | VERIFIED (contains 1 session leakage: 20200730 split 147 train/119 test) | Defines historical split partitions and metadata |
| 7 | **Dataset Manifest (100% Group Session Holdout)** | `outputs/phase52_final_validation/session_holdout_manifest.csv` | 64,163 total rows (train=44,767, val=6,416, test=12,980) | N/A (split definition) | VERIFIED (0 cross-split sessions; all 80 sessions strictly partitioned) | Authoritative session-held-out manifest with 100% group isolation |
| 8 | **Real OOD Features Tensor** | `outputs/casm_v2/phase50_ood_features/ood_real_features.pt` | 5,000 real images (1,000 per class: Cherry, Kiwi, Peach, Pineapple, Plum) | 768-D ConvNeXt-Tiny embeddings | VERIFIED (15.8 MB, sha256 checksummed, real image extraction) | Feature embeddings for open-set detection evaluation on real unseen species |
| 9 | **Real OOD Predictions & Detector Scores** | `outputs/final_audit/ood_predictions.csv` | 5,000 rows with Mahalanobis, Energy, and MSP scores | 3 detectors | VERIFIED (scores matched to test evaluation) | Individual image OOD scores for thresholding and ROC/PR curve computation |
| 10 | **OOD Benchmark Summary Table** | `outputs/casm_v2/phase50_ood_benchmark.csv` | 3 rows (Mahalanobis, Energy, MSP) | 3 detector evaluations | VERIFIED (AUROC 84.22%, 81.99%, 74.26%) | Authoritative pooled OOD detection metrics on 5,000 real images |
| 11 | **Selective Risk-Coverage Exemplar (Seed 83)** | `outputs/casm_v2/phase50_per_seed_predictions/preds_A_seed83.pt` | 12,833 test samples | Seed 83, Config A | VERIFIED (MAE 0.0889 -> 0.0136 at 90% coverage; 84.70% reduction) | Single-model exemplar demonstrating risk-coverage tradeoff |
| 12 | **Ensemble Calibration Artifact** | `outputs/casm_v2/phase50_calibration_report.md` | 12,833 test samples (30-seed probability ensemble) | Config A, 30 seeds ensemble | VERIFIED (ECE = 2.51%, Brier = 0.0887) | Probability calibration of the multi-seed model ensemble |
| 13 | **Hardware Latency & Throughput Benchmark** | `outputs/phase54_discrepancy_resolution/issue5_benchmarks.json` | 12 configuration combinations (Head vs E2E, FP32 vs FP16, BS 1/16/32) | NVIDIA GeForce RTX 3050 Laptop GPU | VERIFIED (FP16 E2E: 49.0 FPS BS=1, 90.4 FPS BS=32; FP32: 27.0 & 40.1) | Empirical runtime benchmark on RTX 3050 with synchronized timing |
| 14 | **Automated Unit & Integration Test Suite** | `tests/test_*.py and outputs/casm_v2/test_corrected_modules.py` | 21 automated unit tests | All components | VERIFIED (21/21 passed in 110.9s under pytest) | Continuous integration test harness for model, data, and metric integrity |

---

## Verification Protocol & Integrity Guarantees

1. **Raw Artifact Immutability:** All 120 historical checkpoint `.pth` files and 120 `.pt` prediction tensors remain intact in their original locations (`outputs/casm_v2/`). No retraining or overwriting occurred.

2. **Strict Session Holdout Isolation:** All 120 replication models were trained on `outputs/phase52_final_validation/session_holdout_manifest.csv` where 100% of the 80 acquisition sessions are strictly confined to a single partition.

3. **Real OOD Image Extraction:** All 5,000 OOD images are real fruits from unencountered species (Cherry, Kiwi, Peach, Pineapple, Plum). Synthetic fallback features have been permanently purged.

4. **Zero Missing Artifacts:** 100% of required model checkpoints, prediction tensors, feature matrices, and evaluation tables exist and have been validated on disk.

