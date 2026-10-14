# Forensic Changelog & Progression Record

**Project:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation  
**Directory:** `./` (Repository Root)  

---

## Phase 5.0 — Multi-Seed Confirmatory Framework
- Formalized 30-seed prespecified evaluation protocol across configurations A, B, C, and D ($N=120$ model runs on Historical Split V2).
- Cached visual representations from frozen ConvNeXt-Tiny backbone (768 dimensions, 56,036 samples).
- First identified negative performance degradation when conditioning visual features with categorical species tags.

## Phase 5.1 — Forensic Integrity & Threshold Inversion Audit
- Discovered critical mathematical flaw in legacy unconstrained CORAL: cutpoint thresholds $b_k$ frequently inverted ($b_k > b_{k-1}$), producing negative discrete probability mass and non-monotonic cumulative logits.
- Developed `MonotonicCORALHead` using softplus parameterization $b_k = \theta_0 - \sum_{j=1}^k \text{softplus}(\delta_j)$, structurally guaranteeing monotonic cutpoints $b_0 > b_1 > \dots > b_{K-2}$ and valid probabilities unconditionally.
- Identified turntable session leakage in historical Split V2: Session `20200730` had 147 samples in training and 119 in test.

## Phase 5.2 — Strict Group Session-Held-Out Protocol
- Sanitized dataset partitioning by moving all 266 samples of Session `20200730` to the test set, creating `session_holdout_manifest.csv` ($N_{\text{test}} = 12,980$).
- Conducted full 30-seed retrain campaign ($N=120$ runs), confirming that the negative effect of species conditioning persists under strict session isolation ($\Delta = -0.0262, p = 0.01018$).

## Phase 5.3 & 5.4 — Discrepancy Resolution & Audit Alignment
- Reconciled session-held-out statistical reporting: clarified that B-A contrast has raw $p = 0.01018$ and Holm-adjusted $p = 0.01018$ (earlier drafted text citing $p=0.0078$ was an exploratory placeholder).
- Re-evaluated OOD recognition on 5,000 real images of unencountered species (*Cherry, Kiwi, Peach, Pineapple, Plum*), retiring synthetic Gaussian noise fallbacks.
- Discovered domain-dependent OOD performance inversion: Mahalanobis distance collapses on natural market bins (18.08% AUROC) while Free Energy Score remains robust (92.49%).

## Phase 5.5 — Hardening & Multiplicity Control
- Applied prespecified Holm-Bonferroni multiplicity control across the 3-contrast family $\{B-A, C-A, D-A\}$.
- Clarified selective prediction scopes: distinguished single Seed 83 exemplar (84.70% MAE reduction) from 30-seed aggregate mean (81.05% MAE reduction) and ensemble ECE (2.51%).
- Parameter census verification: confirmed exactly 626,180 trainable parameters in Monotonic CORAL head.

## Phase 5.6 — Publication Lock & Exact Permutation Verification
- Resolved the historical Wilcoxon discrepancy:
  - Identified exact tie between Seed 3 ($\Delta = +0.0263$) and Seed 73 ($\Delta = -0.0263$) at ranks 13 and 14 (average rank 13.5).
  - Calculated exact positive rank sum $W^+ = 71.5$, negative rank sum $W^- = 393.5$, and signed-rank statistic $W = \min(W^+, W^-) = 71.5$.
  - Discovered that SciPy 1.17's `method='exact'` evaluated $W=72$ against an integer no-ties distribution ($p=0.000555$).
  - Implemented dynamic programming exact conditional permutation test enumerating all $2^{30} = 1,073,741,824$ sign assignments conditioned on observed ranks with ties, proving the true exact conditional p-value is **$p = 0.000514$** ($5.143285 \times 10^{-4}$).
- Tempered broad conclusions: bounded claims strictly to evaluated 30 seeds and experimental split protocols.
- Separated test images $N_{\text{test}}$ from paired training seeds $n_{\text{seeds}} = 30$.
- Qualified hardware benchmarks: noted that 49.0–90.4 FPS on RTX 3050 reflects forward neural inference and does not include physical conveyor camera capture and mechanical ejection.

## Final Submission Package Assembly
- Disentangled and resolved QWK transcription discrepancy: established single machine-readable source of truth `final_results.json` directly from the 240 raw `.pt` prediction tensors:
  - Historical: A ($0.8478 \pm 0.0260$), B ($0.8118 \pm 0.0364$), C ($0.7778 \pm 0.0450$), D ($0.7630 \pm 0.0402$).
  - Session Holdout: A ($0.8038 \pm 0.0335$), B ($0.7776 \pm 0.0431$), C ($0.7513 \pm 0.0352$), D ($0.7164 \pm 0.0494$).
- Corrected significance statements: clearly stated that session-held-out B-A is significant at $\alpha = 0.05$ ($p_{\text{adj}} = 0.025888$), avoiding overbroad claims of $p < 0.001$ across all tests.
- Re-executed all 28 automated tests (exit code 0).
- Generated publication figures at 300 DPI directly from `final_results.json`.
- Packaged complete submission artifacts in `outputs/final_submission/`.

## Phase 5.7 — Final Submission Gap-Closure & Publication Lock
- **Gap 1 (Independent Metric Recomputation):**
  - Independently recomputed QWK, MAE, accuracy, and macro-F1 across all 240 prediction tensors (4 configurations × 30 seeds × 2 protocols).
  - Confirmed 100% target label concordance against split manifests (`phase50_split_v2_manifest.csv` and `session_holdout_manifest.csv`).
  - Confirmed all test counts: Historical Split V2 = 12,833; Strict Group Session Holdout = 12,980; Dataset Total = 56,036.
  - Confirmed session isolation: Session `20200730` overlap in Historical split (147 train, 119 test, 266 total) vs zero session leakage across all 80 sessions in Session Holdout.
  - Certified `verification_status: EXACT_REPRODUCED` for all 240 model runs (max absolute QWK diff against stored = 0.000050, purely due to 4-decimal rounding).
- **Gap 2 (OOD Metric & Score Reconciliation):**
  - Recomputed and harmonized pooled and domain-stratified OOD benchmarks from 5,000 real images (Cherry, Kiwi, Peach, Pineapple, Plum):
    - Mahalanobis Distance: AUROC 84.22%, AUPR (OOD=1) 53.28%, FPR@95 38.39%.
    - Free Energy Score (Historical): AUROC 81.99%, AUPR (OOD=1) 49.96%, FPR@95 41.38%.
    - MSP (Historical): AUROC 74.26%, AUPR (OOD=1) 43.16%, FPR@95 47.85%.
    - Free Energy Score (Corrected symmetric LayerNorm): AUROC 63.03%, AUPR (OOD=1) 38.32%, FPR@95 73.14%.
    - MSP (Corrected symmetric LayerNorm): AUROC 56.27%, AUPR (OOD=1) 32.22%, FPR@95 74.76%.
  - Disentangled historical draft discrepancies (approx. 54.1% / 44.2% and 41.9% / 61.3%) as originating from preliminary unverified draft notes where score arrays were not preserved, confirming that operating threshold calibration cannot explain an AUPR shift. Traced the mathematical cause of the 81.99% / 74.26% benchmark to an asymmetric classifier head normalization between cached ID features and end-to-end OOD forward passes.
  - Synchronized domain-stratified MSP AUROC values (fruits360: 75.21%, real_quality: 61.74%, real_grade: 96.57%, real_severity: 95.30%).
- **Gap 3 (Paired Statistical Tests & Exact Wilcoxon Conditional Permutation):**
  - Verified paired seed matching for all 6 contrasts (B-A, C-A, D-A under both protocols) across all 30 seeds.
  - Disentangled the tie-aware Wilcoxon calculation: demonstrated that rounded 4-decimal QWKs produce a tie between Seed 3 and Seed 73 ($|\Delta| = 0.0263$, mid-rank 13.5, $W=71.5$, exact conditional $p = 0.000514$), while full float64 precision resolves the difference by $2.08 \times 10^{-5}$ ($W=72.0$, integer ranks, exact $p = 0.000555$).
  - Recomputed Holm-Bonferroni adjusted p-values: historical contrasts significant at $\alpha = 0.001$ ($p_{\text{adj}} \le 0.000555$); session-held-out B-A significant at $\alpha = 0.05$ ($p_{\text{adj}} = 0.026229$).
- **Final Integration & Verification:**
  - Full regression harness executed: 28/28 tests passed in 110.14s (Exit Code 0).
  - All artifacts compiled from authoritative results: manuscript, tables, figures, JSON, claim ledger, and SHA-256 manifests.
  - Final package locked in `outputs/phase57_final_submission_lock/` and synchronized to `outputs/final_submission/`.

