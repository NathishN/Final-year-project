# Manuscript Revision Log

**Project:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading  
**Auditor:** Antigravity Senior Machine Learning Research Scientist & Publication Auditor  
**Date:** 2026-10-13  

---

## Revisions Executed

1. **Training Protocol Methodology Disclosed:**
   - Updated `paper/methodology.md` and `paper/manuscript.md` to explicitly state that confirmatory runs were trained on frozen 768-D ConvNeXt-Tiny embeddings using AdamW on the ordinal and fusion heads only (~626k–630k trainable parameters).
   - Clarified that raw image transformations were applied during Stage 1 backbone training, not during the 120-run confirmatory head training.

2. **Statistical Reconciliation Synchronized:**
   - Updated `paper/results.md` and `paper/manuscript.md` to report the exact Wilcoxon signed-rank test statistic ($W = 72.0, p = 0.000555$) computed via SciPy 1.13 (`zero_method='wilcox'`), replacing erroneous historical values ($W = 59$ and $W = 62$).
   - Documented the paired $t$-test ($t(29) = -4.159, p = 0.000260$, Cohen's $d_z = -0.7593$) and Holm-Bonferroni family-wise error rate control ($p_{adj} < 0.001$).

3. **Session-Level Generalization Clarified:**
   - Added Phase 5.2 session holdout findings: Session `20200730` split in `split_v2` was documented, and clean group-held-out re-evaluation (12,980 test samples, 0 cross-split sessions) confirmed that Config A ($0.8266$) outperforms Config B ($0.7980$) under strict group isolation.

4. **Open-Set Domain Confounding Bounded:**
   - Qualified the background shortcut hypothesis in `paper/results.md` and `paper/limitations.md`: explicitly noted that while Mahalanobis collapses to 18.08% on grocery store market bins, Energy score achieves 92.49% AUROC on the same subset.
   - Designated causal background attribution as scientifically UNRESOLVED due to lack of ground-truth foreground segmentation masks in the raw dataset.

5. **Throughput and Latency Disaggregated:**
   - Distinguished head-only throughput (>100,000 FPS on batch sizes $\ge 16$) from full end-to-end convolutional pipeline throughput (41.0 FPS at BS=1, 75.6 FPS at BS=32 in FP16 on NVIDIA RTX 3050).

6. **Commercial Deployment Claims Calibrated:**
   - Removed unhedged assertions of conveyor line deployment readiness from `paper/abstract.md` and `paper/limitations.md`. Positioned findings as initial computational inference benchmarks awaiting operational physical sorting integration.
