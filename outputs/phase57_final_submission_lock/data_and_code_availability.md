# Data and Code Availability Statement

**Manuscript Title:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation  
**Target Journal:** IEEE Transactions on Agri-Food Computing / Pattern Analysis & Machine Intelligence  

---

## 1. Code Availability

All source code required to reproduce the reported preprocessing, model training, evaluation, statistical hypothesis testing, calibration analysis, out-of-distribution detection benchmarks, and publication figures is open-source and included in this submission repository under the MIT License.

- **Core Model Code:** `model.py`, `loss.py`, `outputs/casm_v2/casm_v2_corrected_modules.py`
- **Statistical Evaluation Scripts:** `scripts/generate_authoritative_final_results.py`, `outputs/phase56_publication_lock/compute_exact_permutation_tests.py`
- **OOD Evaluation Harness:** `outputs/phase56_publication_lock/revalidate_ood.py`
- **Figure Generation:** `scripts/generate_final_submission_figures.py`
- **Regression Test Suite:** `tests/test_phase56_regression_hardening.py`

---

## 2. Model Checkpoint Availability

All 240 trained neural network model checkpoints (`.pth` files) and raw per-sample evaluation tensors (`.pt` files) generated during the confirmatory 30-seed campaign are preserved and organized by configuration and seed:
- **Historical Split V2:** `outputs/casm_v2/phase50_per_seed_predictions/` (120 `.pt` files, 30 seeds × 4 configurations)
- **100% Group Session Holdout:** `outputs/phase52_final_validation/session_holdout_retrain/predictions/` (120 `.pt` files, 30 seeds × 4 configurations)
- **Exemplar Checkpoint:** `outputs/casm_v2/phase50_checkpoints/model_A_seed83_best.pth` (verifying the exact 626,180 trainable parameters)

---

## 3. Dataset Manifest Availability and Licenses

Due to redistribution licensing conditions of underlying third-party image collections, raw image files are subject to their original publisher licenses:
1. **Fruits-360 Dataset:** Published under CC BY-SA 4.0 (Mureșan & Oltean, 2018).
2. **Real Fruit Quality Dataset:** Open academic research access via Mendeley Data.
3. **Open-Set Anomaly Benchmark:** 5,000 real images spanning five unseen fruit species (*Cherry, Kiwi, Peach, Pineapple, Plum*) curated for zero-shot open-set evaluation.

To guarantee bitwise reproducibility without violating copyright redistribution terms, complete CSV manifests documenting every image path, normalized species label, severity grade, physical acquisition session ID, and SHA-256 bitwise hash are provided in the repository:
- `outputs/casm_v2/phase50_split_v2_manifest.csv` (56,036 samples)
- `outputs/phase52_final_validation/session_holdout_manifest.csv` (56,036 samples)

Precomputed ConvNeXt-Tiny visual feature tensors (56,036 samples × 768 dimensions) are provided in `outputs/features/phase4_dataset.pt` to permit immediate, full replication of all statistical and evaluation results without requiring expensive image re-extraction.
