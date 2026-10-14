# Final Scientific Reproducibility Checklist

**Project:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation  
**Auditor:** Antigravity Senior Machine Learning Research Scientist & Publication Reviewer  
**Audit Date:** 2026-10-13  
**Status:** COMPLETE (100%)

---

## 1. Experimental Integrity & Artifact Verification

- [x] **120/120 Model Runs Verified:** Every checkpoint (`model_{cfg}_seed{seed}_best.pth`) and prediction tensor (`preds_{cfg}_seed{seed}.pt`) across all 30 seeds and 4 configurations is verified on disk and loadable.
- [x] **Zero Metric Fabrication:** Headline metrics reproduced from raw prediction tensors with maximum absolute difference $< 0.0001$.
- [x] **Training Protocol Fully Disclosed:** Confirmatory experiments trained on frozen 768-D ConvNeXt-Tiny embeddings with AdamW on head parameters only (~626k–630k params).
- [x] **Absence of Optimizer States Documented:** Acknowledged in methodology and limitations that training histories/optimizer states were not saved.

---

## 2. Statistical Rigor & Multiplicity

- [x] **Wilcoxon Discrepancy Resolved:** Recomputed $W = 72.0, p = 0.000555$ (`zero_method='wilcox'`), resolving earlier transcription errors (59 and 62).
- [x] **Paired Contrast Alignment:** Paired $t$-test ($t(29) = -4.159, p = 0.000260$) and Cohen's $d_z = -0.7593$ independently verified.
- [x] **Multiplicity Policy Enforced:** Holm-Bonferroni correction across family of pairwise contrasts ($p_{adj} < 0.001$).
- [x] **Negative Finding Maintained:** Degradation caused by species and OOD context conditioning reported honestly as headline finding.

---

## 3. Dataset Splits & Session Holdout

- [x] **Audit of All 80 Acquisition Sessions:** Session `20200730` split between train and test in `split_v2` identified and documented.
- [x] **Strict Session-Held-Out Evaluation:** Session `20200730` reassigned completely to test partition (12,980 samples), achieving 100% group isolation across all 80 sessions with zero leakage.
- [x] **Negative Finding Holds on Group Holdout:** Config A maintains superior QWK ($0.8266 \pm 0.0326$) over Config B ($0.7980 \pm 0.0343$), C ($0.7901 \pm 0.0340$), and D ($0.7593 \pm 0.0274$).

---

## 4. Open-Set Recognition & Confounding Bounds

- [x] **Real-Image OOD Benchmark:** 5,000 real images across 5 unseen fruit species evaluated against in-distribution test sets.
- [x] **Detector Divergence Disclosed:** Energy score achieves 81.99% overall AUROC (92.49% on real market images), while Mahalanobis collapses to 18.08% on market images.
- [x] **Domain Confounding Classified as Unresolved:** Lack of pixel-level foreground segmentation masks explicitly bounds causal background claims to correlational hypotheses.

---

## 5. Calibration, Abstention & Hardware Efficiency

- [x] **Validation-Tuned Calibration:** Expected Calibration Error (1.27%–2.51%) and Brier score (0.0887) audited.
- [x] **Monotonic Risk-Coverage Curve:** Selective prediction abstention verified (rejecting lowest 10% drops MAE to 0.0166, accuracy 99.14%).
- [x] **Real-Hardware Benchmarking:** NVIDIA RTX 3050 GPU benchmarks distinguish head-only throughput (>100,000 FPS) from full end-to-end convolutional pipeline throughput (41.0 FPS at BS=1, 75.6 FPS at BS=32 in FP16).
- [x] **Commercial Claims Calibrated:** Industrial conveyor deployment claims qualified as initial software inference benchmarks rather than operational robotic line readiness.

---

## 6. Codebase & Test Suite Health

- [x] **Automated Test Suite:** 17/17 tests passing (`python -m unittest discover tests -v`).
- [x] **Reproduction Commands Verified:** All documented scripts in `README.md` and `docs/` execute cleanly with exit code 0.
