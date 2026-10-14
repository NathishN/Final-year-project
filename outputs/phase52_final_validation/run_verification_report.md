# Phase 5.2 — 120-Run Experiment Verification Report

**Date:** 2026-10-13  
**Status:** 120 / 120 runs artifact-verifiable (`VERIFIED_COMPLETE`).

## Summary by Configuration

| Configuration | Expected Runs | Verified Checkpoints | Verified Predictions | QWK Match Rate |
|---|---:|---:|---:|---:|
| A (Image-Only Baseline) | 30 | 30 | 30 | 30/30 |
| B (Image + Species Conditioning) | 30 | 30 | 30 | 30/30 |
| C (Image + OOD Context) | 30 | 30 | 30 | 30/30 |
| D (Image + Species + OOD Context) | 30 | 30 | 30 | 30/30 |

## Methodological Limitations

1. **Absence of Optimizer States:** No optimizer state dicts were persisted in checkpoint `.pth` files. While the learned weights and inference outputs are 100% loadable and verifiable, step-by-step training trajectory replication cannot be verified without retraining.
2. **Absence of Epoch Logs:** Intermediate validation loss per epoch was not saved to disk; model selection was performed during the Phase 5.0 run via in-memory validation QWK tracking.
