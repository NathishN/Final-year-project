# Repository Inventory & Artifact Census

**Audit Timestamp:** 2026-10-13 16:17:37  
**Total Cataloged Files:** 154,028  
**Repository Root:** `C:\fruit_dataset`  

---

## 1. File Distribution by Category

| Category | File Count | Total Size (MB) |
|:---|---:|---:|
| `checkpoint_or_tensor` | 298 | 1,092.64 MB |
| `dataset_or_metrics_csv` | 26 | 17.3 MB |
| `image_data` | 153,493 | 3,674.28 MB |
| `metadata_json` | 10 | 0.05 MB |
| `other` | 77 | 0.03 MB |
| `report_or_documentation` | 67 | 0.28 MB |
| `source_code_python` | 57 | 0.63 MB |

---

## 2. Key Checkpoints and Feature Datasets

| Key Artifact | Category | Size (KB) | SHA-256 (Short) | Provenance / Role |
|:---|:---|---:|:---:|:---|
| `master_dataset.csv` | Dataset | 10217.23 | f26801989d0ba9db | Master dataset manifest containing split_v1 annotations |
| `outputs/features/phase4_dataset.pt` | Features | 175031.83 | 8b83725e70d72aea | Pre-extracted 768-D ConvNeXt-Tiny embeddings for all 56,036 samples |
| `outputs/checkpoints/best_model.pth` | Backbone Checkpoint | 326285.33 | SKIPPED_LARGE_OR_IMAGE | Phase 1 ConvNeXt-Tiny species classification backbone |
| `outputs/casm_v2/phase50_ood_features/ood_real_features.pt` | Real OOD Features | 15503.52 | 906fe978a5a7b5b7 | 5,000 real unencountered fruit features from open-set-dataset |
| `outputs/casm_v2/phase50_split_v2_manifest.csv` | Split Manifest | 6067.21 | 1a3543136739385f | Frozen split_v2 allocating session 20200730 to test (119 Grade 2 samples) |

---

## 3. Verified Confirmatory Checkpoint & Prediction Artifacts
- **Phase 5.0 Checkpoints:** 120 `.pth` model state files in `outputs/casm_v2/phase50_checkpoints/` (4 configurations × 30 seeds).
- **Phase 5.0 Per-Sample Predictions:** 120 `.pt` prediction tensors in `outputs/casm_v2/phase50_per_seed_predictions/`.