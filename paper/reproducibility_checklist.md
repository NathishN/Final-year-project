# Reproducibility Checklist

- [x] **Frozen Split Manifest:** All 56,036 sample assignments persisted in `outputs/final_audit/split_manifest.csv`.
- [x] **Deterministic Seeds:** Exact schedule of 30 prespecified seeds documented in `configs/training_protocol.json`.
- [x] **Zero Test-Set Hyperparameter Tuning:** All checkpoint selections performed strictly on validation QWK/MAE.
- [x] **100% Artifact Persistence:** All 120 model checkpoints (`.pth`) and raw per-sample prediction tensors (`.pt`) saved in `outputs/casm_v2/`.
- [x] **Exact Numerical Reproduction:** Recalculated metrics match reported values with 0.000000 maximum error.
- [x] **Automated Test Suite:** 17 unit, property, and integration tests pass with 100% success (`tests/`), including full verification of saved prediction tensors and statistical tests.
- [x] **Benchmarked Pipeline Throughput:** Both head-only (>120k FPS) and full end-to-end (41–76 FPS in FP16 on RTX 3050) latencies empirically measured.
