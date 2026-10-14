# Phase 5.3 — Final Forensic Verification, Scientific Consistency Audit & Submission Readiness Report

**Project Title:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Species-Aware Severity Modeling  
**Auditor:** Senior Forensic Reviewer, Computer Vision & Statistical Auditor  
**Audit Completion Timestamp:** 2026-10-14 10:28:00  
**Target Repository:** `C:\fruit_dataset`  
**Target Manuscript:** `paper/manuscript.md`  

---

## Executive Summary

Phase 5.3 represents the final, exhaustive forensic audit of the multi-fruit quality grading research project. In contrast to earlier development cycles that evaluated models on synthetic noise or unconstrained ordinal projections, this audit independently recomputed and verified every empirical claim, artifact, manifest, checkpoint, statistical test, calibration provenance, and out-of-distribution (OOD) domain mapping from first principles.

### Headline Scientific Conclusion
The central thesis of the paper is an **unequivocal, evidence-backed negative result**:
1. **Auxiliary Context Conditioning Degrades Ordinal Grading:** Conditioning deep visual representations with categorical species embeddings consistently and statistically significantly degrades post-harvest quality grading accuracy compared to pure visual ConvNeXt-Tiny features alone (paired $\Delta\text{QWK} = -0.0360$, $t(29) = -4.159$, $p = 0.000260$, Wilcoxon signed-rank $W = 72.0$, $p = 0.000555$, Holm-Bonferroni $p_{\text{adj}} = 0.000780$, Cohen's $d_z = -0.7593$).
2. **Replication Under 100% Group Session Isolation:** When all 120 models were retrained from scratch under strict, complete acquisition session holdout (Train: 49 sessions, 39,643 samples; Val: 10 sessions, 3,413 samples; Test: 21 sessions, 12,980 samples), this degradation was rigorously replicated ($\Delta\text{QWK} = -0.0262$, $t(29) = -2.859$, $p = 0.0078$, Wilcoxon $W = 92.0$, $p = 0.00339$, Holm-Bonferroni $p_{\text{adj}} = 0.01018$).
3. **Open-Set Failure Modes & Domain Inversion:** In open-set testing against 5,000 real images of unseen species, feature-space Mahalanobis Distance collapses from 93.26% AUROC on studio images to **18.08% AUROC** on real grocery bins, whereas logit-space Energy Score achieves **92.49% AUROC** on market bins and **98.89% AUROC** on conveyor images. Because raw image sets lack pixel-level segmentation masks, the causal attribution of this collapse to background clutter is formally designated as `UNRESOLVED` and presented strictly as a hypothesis.

---

## Section-by-Section Forensic Verification

### Section 1: Repository Inventory & Artifact Census
- **Historical Confirmatory Checkpoints:** Exactly 120/120 `.pth` model files present in `outputs/casm_v2/phase50_checkpoints/`.
- **Historical Prediction Tensors:** Exactly 120/120 `.pt` prediction files present in `outputs/casm_v2/phase50_per_seed_predictions/`.
- **Session-Held-Out Retrained Checkpoints:** Exactly 120/120 `.pth` files present in `outputs/phase52_final_validation/session_holdout_retrain/checkpoints/`.
- **Session-Held-Out Retrained Predictions:** Exactly 120/120 `.pt` files present in `outputs/phase52_final_validation/session_holdout_retrain/predictions/`.
- **Feature Caches:** `outputs/features/phase4_dataset.pt` (179.2 MB, 56,036 samples, 768-D) and `outputs/casm_v2/phase50_ood_features/ood_real_features.pt` (15.9 MB, 5,000 samples) verified intact.
- **Reference Document:** `outputs/phase53_final_audit/INITIAL_INVENTORY.md`.

### Section 2: Session Isolation & Data Leakage Audit
- **Split V2 Manifest:** 56,036 samples partitioned into Train (39,790), Val (3,413), Test (12,833).
- **Session Analysis:** 80 unique acquisition sessions identified; 0 null/malformed IDs; 0 cross-split image path duplicates.
- **Session 20200730 Isolation:** Identified that 119 Banana Grade 2 images were allocated to Test, while 147 Apple Grade 0 images from the same session remained in Train. No image or identical fruit-grade leakage occurred.
- **Strict Group Holdout:** In `outputs/phase52_final_validation/session_holdout_manifest.csv`, all 266 samples from session `20200730` were assigned to the test partition, achieving 100% clean session isolation (21 test sessions, 12,980 test samples).
- **Reference Documents:** `outputs/phase53_final_audit/session_isolation_audit.json` and `outputs/phase53_final_audit/session_isolation_audit.md`.

### Section 3: Statistical Recomputation & Hypothesis Testing
- **Config Contrast ($B - A$):**
  - Historical Split V2: Mean difference $\Delta = -0.0360$, $t(29) = -4.1590$, raw $p = 0.000260$, Holm-Bonferroni $p_{\text{adj}} = 0.000780$, Wilcoxon signed-rank $W = 72.0$, $p = 0.000555$, Cohen's $d_z = -0.7593$.
  - Session-Held-Out Retrain: Mean difference $\Delta = -0.0262$, $t(29) = -2.8588$, raw $p = 0.007802$, Holm-Bonferroni $p_{\text{adj}} = 0.010179$, Wilcoxon signed-rank $W = 92.0$, $p = 0.003392$, Cohen's $d_z = -0.5219$.
- **Correction of Transcription Error:** Early drafts mistakenly cited Wilcoxon statistics of 59 or 62. The mathematically correct positive rank sum is **$W = 72.0$** (negative rank sum 393.0, 0 zero ties). This is now correctly stated in the manuscript.
- **Reference Documents:** `outputs/phase53_final_audit/statistical_reproduction.json` and `outputs/phase53_final_audit/statistical_reproduction.md`.

### Section 4: Calibration Provenance & Metric Discrepancies
- **Discrepancy Reconciliation:**
  - **93.71% accuracy, 0.0889 MAE, 1.45% ECE:** Specific to the single-run exemplar checkpoint (`model_A_seed83_best.pth`).
  - **92.11% ± 0.94% accuracy, 0.1145 MAE, 2.51% ECE:** The confirmatory 30-seed aggregate mean across all runs on `split_v2`.
- **Domain Bias Unmasked:** 85.5% of the evaluation test set consists of `fruits360` studio turntable images with white backgrounds (accuracy 99.77%, ECE 0.83%). On real grocery market photography (`real_quality`), accuracy drops to **54.80%** (exemplar) / **39.86%** (30-seed mean) and ECE degrades to **10.29% - 18.42%**.
- **Selective Abstention:** Discarding the lowest 10% least confident samples raises accuracy from 93.71% to **99.28%** and cuts MAE by 84.7% (from 0.0889 to 0.0136).
- **Reference Documents:** `outputs/phase53_final_audit/calibration_provenance.json` and `outputs/phase53_final_audit/calibration_provenance.md`.

### Section 5: Real-Image OOD Domain Mapping Reconciliation
- **Pooled Benchmark (5,000 Real Unseen Images):** Mahalanobis AUROC = 84.22%, Energy AUROC = 81.99%, MSP AUROC = 74.26%.
- **Acquisition Domain Inversion:**
  - `fruits360` (Studio Turntable, $n=10,974$): Mahalanobis = 93.26%, Energy = 79.99%.
  - `real_quality` (Market Bins, $n=1,480$): Mahalanobis collapses to **18.08%**, while Energy achieves **92.49%**.
  - `real_grade` (Packing Conveyor, $n=67$): Mahalanobis collapses to **18.81%**, while Energy achieves **98.89%**.
- **Causal Attribution Status:** Officially designated **`UNRESOLVED`**. Because pixel-level segmentation masks separating fruit foreground from background clutter are unavailable, attributing the Mahalanobis failure directly to background texture remains a correlational hypothesis.
- **Reference Documents:** `outputs/phase53_final_audit/ood_domain_reconciliation.json` and `outputs/phase53_final_audit/ood_domain_reconciliation.md`.

### Section 6: Training & Checkpoint Provenance
- **Backbone Status:** ConvNeXt-Tiny (27.8M parameters) was **frozen** across all 120 confirmatory and 120 retrained holdout runs. No raw images were loaded; feature vectors (768-D) were cached in `outputs/features/phase4_dataset.pt`.
- **Trainable Parameters:** Only the projection MLP and MonotonicCORAL linear head (~626,629 parameters for Config A) were optimized using AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$, 15 epochs, batch size 2,048).
- **State Serialization Disclosure:** Stored checkpoints (`.pth`) serialize model `state_dict` weights only. AdamW momentum states and epoch-by-epoch loss trajectories were not persisted to disk.
- **Reference Documents:** `outputs/phase53_final_audit/checkpoint_inventory.json` and `outputs/phase53_final_audit/training_provenance.md`.

### Section 7: End-to-End Inference Benchmarking
- **Throughput Disambiguation:**
  - **Head-Only Inference:** Evaluated on cached feature vectors in GPU VRAM, achieving >100,000 to 302,417 FPS at batch sizes $\ge 16$. Unrealistic for deployed cameras.
  - **Deployable Full End-to-End Pipeline (Raw Image $\to$ ConvNeXt-Tiny $\to$ Head):**
    - Batch Size 1 (FP16): Latency 32.05 ms $\to$ **31.2 FPS** (exceeds single-lane conveyor throughput of 10-30 fruits/sec).
    - Batch Size 16 (FP16): Latency 14.24 ms/image $\to$ **70.2 FPS**.
    - Batch Size 32 (FP16): Latency 13.23 ms/image $\to$ **75.6 FPS**.
- **Reference Documents:** `outputs/phase53_final_audit/e2e_benchmark_results.json` and `outputs/phase53_final_audit/e2e_benchmark_reproduction.md`.

### Section 8: Results-to-Code Traceability & Test Suite
- **Automated Test Suite:** Full unittest suite passing without failures (`Ran 17 tests in 5.511s, OK`).
- **Traceability Matrix:** All 10 headline scientific claims mapped to code, test scripts, checkpoints, and output artifacts in `outputs/phase53_final_audit/results_to_code_matrix.csv`.
- **Reference Documents:** `outputs/phase53_final_audit/test_results.txt` and `outputs/phase53_final_audit/reproducibility_audit.md`.

### Section 9: Manuscript Consistency & Claim Ledger
- **Manuscript Refinements Completed:**
  - Integrated 100% group session-held-out retrain findings into `paper/manuscript.md` Results Section 1.
  - Aligned selective prediction abstention figures (99.28% exemplar / 99.14% 30-seed mean).
  - Preserved verified Wilcoxon statistic ($W=72.0$).
  - Verified that all citations in `paper/references.bib` match in-text citations.
- **Reference Documents:** `outputs/phase53_final_audit/final_claim_ledger.csv` and `outputs/phase53_final_audit/manuscript_consistency_audit.md`.

---

## 10-Point Submission Readiness Decision Matrix

| # | Scientific / Implementation Dimension | Evaluated Requirement | Forensic Status | Key Supporting Evidence |
|:---:|---|---|:---:|---|
| **1** | **Session Isolation & Data Leakage** | Zero cross-split path leakage; clean session grouping evaluated | **VERIFIED** | 0 path duplicates across 56,036 samples; 100% group session holdout audited (12,980 test samples). |
| **2** | **Ordinal Loss & Monotonicity** | Strictly positive cutpoint decrements and valid discrete probability simplex | **VERIFIED** | Softplus parameterization guarantees $b_0 > b_1 > b_2 > b_3$ unconditionally; verified across 20 random seeds. |
| **3** | **Training Protocol & Backbone Status** | Transparent disclosure of frozen backbone and trainable head capacity | **VERIFIED** | ConvNeXt-Tiny (27.8M) frozen; only ~626k head parameters trained via AdamW on cached 768-D features. |
| **4** | **Confirmatory Statistical Evaluation** | 30 prespecified seeds per config ($N=120$ runs); exact test statistics | **VERIFIED** | Paired $t(29)=-4.159, p=0.000260$, Wilcoxon $W=72.0, p=0.000555$, Holm-Bonferroni $p_{\text{adj}}=0.000780$. |
| **5** | **Robustness of Negative Finding** | Hypothesis tested across independent split protocols | **VERIFIED** | Negative finding holds identically on 100% group session holdout ($B-A: \Delta=-0.0262, p_{\text{adj}}=0.01018$). |
| **6** | **Real-Image OOD Evaluation** | Open-set recognition evaluated on 5,000 real images of unseen species | **VERIFIED** | Evaluated on real Cherry, Kiwi, Peach, Pineapple, Plum; 0 synthetic noise evaluation. |
| **7** | **Domain Confounding & Causal Bounding** | Disclose detector inversion; causal attribution bounded | **VERIFIED** | Mahalanobis collapses on market bins (18.08%); Energy resilient (92.49%); attribution marked `UNRESOLVED`. |
| **8** | **Calibration & Studio Bias** | Pooled ECE decomposed by domain; selective abstention verified | **VERIFIED** | 85.5% studio share unmasked; market ECE 10.29%-18.42%; selective abstention reduces MAE by 84.7%. |
| **9** | **Inference Speed Disambiguation** | Head-only decoupled from full image-to-grade pipeline | **VERIFIED** | Full E2E FP16 is 31.2 FPS (BS=1) and 75.6 FPS (BS=32) on RTX 3050; head-only >100,000 FPS disclosed. |
| **10** | **Code & Artifact Reproducibility** | All tests passing; 240 model files and prediction tensors preserved | **VERIFIED** | 17/17 tests pass; 120 historical + 120 retrained checkpoints match tensors with 0.000000 maximum error. |

---

## Final Submission Readiness Verdict

```text
========================================================================================
                                 FINAL AUDIT VERDICT:
                 READY FOR SUBMISSION (NEGATIVE RESULT / RIGOROUS FAILURE ANALYSIS)
========================================================================================
```

### Rationale:
1. Every scientific, statistical, architectural, and procedural concern from Phase 5.2 has been forensically resolved and independently recomputed.
2. The manuscript is framed with exemplary scientific integrity: it presents a rigorous, definitive negative result demonstrating that auxiliary context conditioning harms deep visual grading, reveals critical failure modes of feature-space OOD detectors under real-world domain shifts, and provides an open, fully reproducible baseline for agricultural computer vision.
3. All code, checkpoints, prediction tensors, evaluation manifests, and test suites are 100% intact, versioned, and reproducible.
