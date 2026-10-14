# Phase 5.1 Forensic Audit — PROGRESS LOG

**Started:** 2026-10-13 16:35 IST  
**Completed:** 2026-10-13 17:22 IST  
**Auditor:** Antigravity (Senior ML Research Auditor & Scientific Reviewer)  
**Project:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading  
**Project Root:** `C:\fruit_dataset`

---

## Status Summary

| Task | Description | Status | Resolution |
|:---:|:---|:---:|:---|
| **A** | Repository & Artifact Inventory | **COMPLETE** | 154,000+ files inventoried, 120 checkpoints and 120 prediction tensors verified |
| **B** | 120-Run Verification Matrix | **COMPLETE** | 120/120 runs artifact-verified; shape and finiteness checked |
| **C** | Training Methodology Audit | **COMPLETE** | Frozen backbone / precomputed feature protocol disclosed in methodology |
| **D** | Recompute Confirmatory Metrics | **COMPLETE** | QWK, MAE, accuracy reproduced across all 120 runs within < 0.0001 error |
| **E** | Statistical Reconciliation (Wilcoxon W) | **COMPLETE** | Reconciled W=71.5 (W=72.0 rounded, zero_method='wilcox', p=0.000555); t-test confirmed |
| **F** | Dataset Split / Session Exclusivity | **COMPLETE** | Clarified Grade 2 Banana session exclusivity vs full session holdout |
| **G** | OOD Metric Audit & Causal Breakdown | **COMPLETE** | Domain breakdown shows Mahalanobis collapse (18.08%) is detector-specific; Energy (92.49%) is robust; background causality labeled as hypothesis |
| **H** | Calibration & Selective Prediction Audit | **COMPLETE** | ECE clarified as seed-dependent and studio-dominated; abstention filtering verified |
| **I** | Efficiency Claims & Throughput Audit | **COMPLETE** | Head-only (>120k FPS) vs Full E2E (41.0 to 76.2 FPS on RTX 3050 FP16) benchmarked and distinguished |
| **J** | Reproduction Command Audit | **COMPLETE** | README commands corrected to match actual script filenames |
| **K** | Test Scope Audit + Integration Tests | **COMPLETE** | Added `tests/test_integration_predictions.py`; all 17 automated tests passing |
| **L** | Manuscript Correction | **COMPLETE** | Manuscript and paper package updated via `scripts/assemble_paper.py` |
| **M** | Final Forensic Verdict | **COMPLETE** | Updated to `PUBLICATION READY — REPAIRS COMPLETE` |

---

## Findings & Resolutions Log

1. **Precomputed Features Protocol (Task C):** Confirmed training operated on precomputed ConvNeXt-Tiny embeddings in `phase4_dataset.pt` with a frozen visual backbone. Disclosed explicitly in `paper/methodology.md` (§4.1), `paper/abstract.md`, and `README.md`.
2. **Wilcoxon Statistic (Task E):** Corrected historical reporting of W=59 and W=62. Recomputed exact Wilcoxon signed-rank test yields $W=71.5$ ($W=72.0$ rounded, `zero_method='wilcox'`), $p=0.000555$. Directional conclusion (significant degradation under context conditioning) remains robustly supported.
3. **Session Exclusivity (Task F):** Qualified that acquisition session `20200730` Banana Grade 2 images ($n=119$) are held out in the test split, while Apple Grade 0 images ($n=147$) from that session remain in training, ensuring within-species/grade exclusivity rather than full session holdout.
4. **OOD Failure Specificity (Task G):** Replaced universal OOD collapse claim with detector-specific finding: Mahalanobis distance drops to 18.08% on market bins, while Energy Score achieves 92.49% on market bins and 98.89% on packing conveyors. Causal attribution to background is labeled as a hypothesis pending foreground masking.
5. **Inference Throughput (Task I):** Disclosed that >100,000 FPS is head-only processing on feature tensors. Benchmarked full end-to-end pipeline (ConvNeXt-Tiny + Monotonic CORAL head) at 41.0 FPS (batch 1) to 76.2 FPS (batch 32) in FP16 on NVIDIA RTX 3050 GPU, satisfying commercial sorting rates.
6. **Reproduction Commands & Code Integrity (Task J & K):** Corrected script names in `README.md`. Added missing `import numpy as np` in `scripts/inference_pipeline.py`. Added `tests/test_integration_predictions.py` loading real saved prediction tensors and asserting QWK match and statistical reconciliation. Full suite passes with 17/17 tests.
