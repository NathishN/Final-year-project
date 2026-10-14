# Phase 5.6 — Final Publication Forensic Audit & Lock Certification

**Audit Authority:** Senior ML Research Engineer, Statistical Auditor & Reproducibility Specialist  
**Project:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation  
**Repository:** Repository Root (`./`)  
**Audit Timestamp:** 2026-10-14 14:13:48  
**Target Venue:** IEEE Transactions on Agri-Food Computing / Pattern Analysis & Machine Intelligence  

---

## 1. Final Publication-Readiness Verdict

# **READY FOR SUBMISSION**

All material numerical claims, statistical discrepancies, split isolations, OOD metrics, calibration definitions, parameter counts, hardware benchmarks, and regression tests have been forensically audited and verified against immutable underlying artifacts.

Zero unverified claims remain in the authoritative scientific ledger.

---

## 2. Forensic Resolution of Critical Audit Items

| Audit Focus Area | Historical Discrepancy / Concern | Forensic Resolution & Authoritative Result | Status |
|:---|:---|:---|:---:|
| **Wilcoxon Statistic** | Conflicting $W=71.5$ vs $72.0$ | Seed 3 and Seed 73 tie at absolute difference $0.0263$ (ranks 13 and 14). Standard tie averaging assigns rank $13.5$ to each, yielding positive rank sum $W^+ = 71.5$ and negative rank sum $W^- = 393.5$. We explicitly define $W = \min(W^+, W^-) = 71.5$. $W=72.0$ was an integer rounding artifact. | **VERIFIED** |
| **Wilcoxon $p$-Value** | Conflicting $p=0.000555$ vs $p=0.000928$ | Under the exact dynamic programming conditional permutation test conditioned on observed ranks (with the tie at $13.5$), exactly $276,128$ sign assignments out of $2^{30}$ yield $W^+ \le 71.5$, giving exact two-sided $p = 5.143285 \times 10^{-4} \approx \mathbf{0.000514}$. SciPy 1.17's `method='exact'` looked up $W=72$ in the integer no-ties distribution ($p = 0.000555$); asymptotic normal approximation yields $p = 0.000928$. | **VERIFIED** |
| **Multiplicity Control** | Potential conflation of raw vs adjusted $p$-values | Holm–Bonferroni step-down correction applied across the prespecified comparison family $\mathcal{F} = \{B-A, C-A, D-A\}$ ($m=3$). All contrasts remain significant ($p_{\text{adj}} < 0.05$). | **VERIFIED** |
| **Sample Size Disambiguation** | Conflation of test image count ($N$) and seed count ($n$) | Explicitly separated across all tables: $N_{\text{test}} = 12,833$ (Historical) and $N_{\text{test}} = 12,980$ (Session Holdout) denote test-set image counts; $n_{\text{seeds}} = 30$ denotes independent model-training seeds (degrees of freedom $\text{df} = 29$). | **VERIFIED** |
| **Selective MAE Reduction** | Arithmetic confusion between 0.0889, 0.1234, and 84.7% reduction | $0.0889$ is the ordinal cutpoint baseline MAE (matching saved checkpoint); $0.1234$ is categorical argmax MAE. $(0.0889 - 0.0136)/0.0889 = 84.70\%$. Ordinal cutpoint is canonical for MonotonicCORAL. | **VERIFIED** |
| **ECE Provenance** | Conflicting ECE values: 1.47%, 2.11%, 2.51% | Fully disentangled across scopes: Seed 83 exemplar ECE is **1.47%**; 30-seed aggregate mean is **2.11% ± 0.53%**; 30-seed probability ensemble is **2.51%**. | **VERIFIED** |
| **Dataset Splits & Session Isolation** | Historical session leakage in session `20200730` | Verified: `20200730` split 147 train / 119 test in `split_v2`. Reallocated in `session_holdout_manifest.csv` so all 266 samples are in test ($N_{\text{test}}=12,980$, train $N=39,643$). Zero session leakage across all 80 sessions. | **VERIFIED** |
| **Real OOD Evaluation** | Previous synthetic 100% AUROC claim | Purged synthetic features. Verified on 5,000 real images across 5 unseen species: Mahalanobis AUROC **84.22%**, Free Energy **81.99%**, MSP **74.26%**. | **VERIFIED** |
| **Domain Detector Inversion** | Mahalanobis collapse to 18.08% on market bins | Verified: Covariance distorts under complex market backgrounds relative to 85.5% studio training set. Free Energy remains robust ($>92\%$). Bounded as an empirical finding without claiming unproven background causality. | **VERIFIED** |
| **Model Parameter Census** | Approximate 626k parameters | Verified layer-by-layer: exactly **626,180 trainable parameters** in MonotonicCORAL head; 27,820,128 frozen parameters in ConvNeXt-Tiny. | **VERIFIED** |
| **GPU Inference Latency** | Milliseconds per batch vs sample ambiguity | Synchronized CUDA benchmark on RTX 3050 Laptop GPU: Full E2E FP16 is **49.0 FPS** (20.43 ms) at BS=1 and **90.4 FPS** (11.07 ms/sample) at BS=32, exceeding mechanical sorting line rates (10–30 fruits/sec). Commercial deployment qualification noted. | **VERIFIED** |
| **Regression Harness** | Test suite completeness | 28 automated unit, property, and regression tests pass with **100% success rate (28/28 passed in 115.36s)** under pytest. | **VERIFIED** |

---

## 3. Authoritative Master Numerical Registry

The following table provides the locked, authoritative reference values for all publications, figures, and presentation materials:

| Parameter / Metric | Authoritative Locked Value | Underlying Primary Artifact |
|:---|:---:|:---|
| **Total Dataset Images** | 56,036 | `outputs/casm_v2/phase50_split_v2_manifest.csv` |
| **Historical Test Samples ($N_{\text{test}}$)** | 12,833 | `outputs/casm_v2/phase50_split_v2_manifest.csv` |
| **Session-Held-Out Test Samples ($N_{\text{test}}$)** | 12,980 | `outputs/phase52_final_validation/session_holdout_manifest.csv` |
| **Unique Acquisition Sessions** | 80 sessions (49 train, 10 val, 21 test) | `outputs/phase52_final_validation/session_holdout_manifest.csv` |
| **Trainable Head Parameters** | 626,180 | `outputs/casm_v2/phase50_checkpoints/model_A_seed83_best.pth` |
| **Frozen Backbone Parameters** | 27,820,128 | Pretrained ConvNeXt-Tiny (ImageNet-1k) |
| **Historical Config A Baseline QWK** | 0.8478 ± 0.0260 | `outputs/casm_v2/phase50_per_seed_predictions/preds_A_seed*.pt` |
| **Historical Config B vs A $\Delta$ QWK** | -0.0360 (95% CI [-0.0537, -0.0183]) | `outputs/phase56_publication_lock/statistics_recomputed.json` |
| **Historical Paired $t(29)$ Statistic** | $t = -4.1590, \; p = 0.000259$ | `outputs/phase56_publication_lock/statistics_recomputed.csv` |
| **Historical Wilcoxon Statistic $W$** | $W = \min(W^+, W^-) = 71.5$ ($W^+=71.5, W^-=393.5$) | `outputs/phase56_publication_lock/statistics_recomputed.csv` |
| **Historical Wilcoxon Exact Permutation $p$** | $\mathbf{p = 0.000514}$ ($5.143285 \times 10^{-4}$) | `outputs/phase56_publication_lock/statistics_recomputed.csv` |
| **Historical Wilcoxon SciPy No-Ties Lookup $p$** | $p = 0.000555$ | `outputs/phase56_publication_lock/statistics_recomputed.csv` |
| **Historical Wilcoxon Asymptotic $p$** | $p = 0.000928$ | `outputs/phase56_publication_lock/statistics_recomputed.csv` |
| **Session-Held-Out Config A QWK** | 0.8175 ± 0.0354 | `outputs/phase52_final_validation/session_holdout_retrain/per_seed_results.csv` |
| **Session-Held-Out B vs A $\Delta$ QWK** | -0.0262 (95% CI [-0.0456, -0.0067]) | `outputs/phase56_publication_lock/statistics_recomputed.csv` |
| **Session-Held-Out Paired $t(29)$** | $t = -2.7490, \; p = 0.01018$ | `outputs/phase56_publication_lock/statistics_recomputed.csv` |
| **Session-Held-Out Wilcoxon $W$** | $W = \min(W^+, W^-) = 125.0$ ($W^+=125.0, W^-=340.0$) | `outputs/phase56_publication_lock/statistics_recomputed.csv` |
| **Session-Held-Out Wilcoxon Exact Perm $p$** | $\mathbf{p = 0.025888}$ | `outputs/phase56_publication_lock/statistics_recomputed.csv` |
| **Real OOD Sample Count** | 5,000 images (5 unseen species $\times$ 1,000) | `outputs/casm_v2/phase50_ood_features/ood_real_features.pt` |
| **Pooled Mahalanobis AUROC** | 84.22% (95% CI [83.7%, 84.8%]) | `outputs/casm_v2/phase50_ood_benchmark.csv` |
| **Pooled Free Energy AUROC** | 81.99% (95% CI [81.3%, 82.6%]) | `outputs/casm_v2/phase50_ood_benchmark.csv` |
| **Pooled MSP AUROC** | 74.26% (95% CI [73.6%, 74.9%]) | `outputs/casm_v2/phase50_ood_benchmark.csv` |
| **Market-Bin Mahalanobis AUROC** | 18.08% | `outputs/casm_v2/phase50_ood_report.md` |
| **Market-Bin Free Energy AUROC** | 92.49% | `outputs/casm_v2/phase50_ood_report.md` |
| **Seed 83 Baseline Cutpoint MAE** | 0.0889 (Accuracy = 93.71%) | `outputs/casm_v2/phase50_per_seed_predictions/preds_A_seed83.pt` |
| **Seed 83 Selective 90% MAE** | 0.0136 (Accuracy = 99.28%) | `outputs/phase56_publication_lock/selective_prediction_recomputed.csv` |
| **Seed 83 Relative MAE Reduction** | **84.70%** (exact) | `outputs/phase56_publication_lock/selective_prediction_recomputed.csv` |
| **30-Seed Aggregate Mean ECE** | 2.11% ± 0.53% | `outputs/phase56_publication_lock/calibration_recomputed.csv` |
| **30-Seed Probability Ensemble ECE**| 2.51% (Brier = 0.0887) | `outputs/casm_v2/phase50_calibration_report.md` |
| **Full Pipeline Throughput (FP16)** | 49.0 FPS (BS=1), 90.4 FPS (BS=32) | `outputs/phase54_discrepancy_resolution/issue5_benchmarks.json` |
| **Automated Unit Tests** | 28 / 28 Passed (0 Failed, 0 Skipped) | `outputs/phase56_publication_lock/test_results.txt` |

---

## 4. Final Sign-Off

The research project is hereby **LOCKED** for publication. All underlying artifacts and scripts are traceable, mathematically verified, and reproducible. The submission package in `outputs/phase56_publication_lock/submission_package/` constitutes a self-contained, reproducible, peer-review-ready submission artifact.
