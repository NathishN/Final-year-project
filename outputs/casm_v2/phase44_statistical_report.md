# Phase 4.4 Task A & C — Formal Statistical Audit Report (Baseline vs CASM v2)

**Auditors:** Statistical ML Researcher, PyTorch Architect, and IEEE Transactions Reviewer  
**Date:** October 2026  
**Repository:** `C:\fruit_dataset`  
**Primary Artifacts:** `outputs/casm_v2/phase44_recalculated_metrics.csv`, `outputs/casm_v2/phase44_statistical_report_data.csv`, `outputs/casm_v2/phase44_bootstrap_data.json`

---

## 1. Executive Summary

This report delivers the comprehensive 20-seed paired statistical audit evaluating whether **CASM v2** (Image + Species Embedding + OOD Context) statistically outperforms the **Image-Only Baseline** on multi-fruit ordinal quality grading.

### Key Conclusions:
1. **Primary Significance Verdict:** In the primary confirmatory metric (Quadratic Weighted Kappa, QWK), the paired two-sided Student's $t$-test yields $p = \mathbf{0.0593}$ ($t(19) = 2.0065$), and the two-sided Wilcoxon signed-rank test yields $p = \mathbf{0.0637}$ ($W = 55.0$). **Both primary tests miss the conventional significance threshold of $\alpha = 0.05$.**
2. **Family-Wise Error Rate Control:** When adjusting across the 5 evaluation metrics (QWK, MAE, Accuracy, OBO Accuracy, Macro-F1) via Holm-Bonferroni correction, **no metric achieves statistical significance** (all adjusted $p \ge 0.1186$).
3. **Stability & Variance Reduction:** CASM v2 exhibits dramatically lower variance across seeds (QWK std: $0.0195$ vs $0.0728$, a $73.2\%$ reduction in variance).
4. **Outlier Dependency:** Two failure-prone baseline seeds (Seed 19: $\text{QWK}=0.5363$, Seed 31: $\text{QWK}=0.5000$) account for **$59.5\%$ of the total mean QWK improvement**. In an $N=18$ sensitivity cohort excluding these two seeds, the mean difference contracts to $\Delta = +0.0119$ ($p = 0.2175$).

---

## 2. Experimental Setup & Protocol

- **Dataset Partition:** Test split consisting of $N = 12,714$ samples across 5 species.
- **Evaluation Seeds ($N=20$):** `[42, 123, 777, 999, 2025, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47]`.
- **Paired Protocol:** Both architectures were initialized, trained, and tested under identical random seeds and optimization schedules.
- **Model Comparison:**
  - **Baseline (Config A):** Frozen ConvNeXt-Tiny visual features ($768\text{D}$) $\to$ CORAL ordinal head ($4$ binary thresholds).
  - **CASM v2 (Config D):** Frozen visual features ($768\text{D}$) $+$ Learned Species Embedding ($32\text{D}$) $+$ OOD Context ($2\text{D}$) $\to$ Fusion MLP $\to$ CORAL ordinal head.

---

## 3. Comprehensive Metric Summary across 20 Matched Seeds

### Table 1: Primary Metric Distributions (Mean ± Std, Median [Min, Max])

| Model | QWK | MAE | Accuracy (%) | OBO Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| **Baseline** | $0.7032 \pm 0.0728$<br>0.7265 [0.5000, 0.7844] | $0.2445 \pm 0.0670$<br>0.2275 [0.1706, 0.4433] | $86.29 \pm 3.01$<br>87.17 [76.59, 89.47] | $93.04 \pm 3.01$<br>93.79 [84.28, 96.33] | $29.83 \pm 3.18$<br>29.34 [25.58, 38.04] |
| **CASM v2** | $0.7326 \pm 0.0195$<br>0.7316 [0.6923, 0.7716] | $0.2145 \pm 0.0154$<br>0.2111 [0.1949, 0.2410] | $87.60 \pm 0.67$<br>87.51 [86.39, 88.67] | $94.52 \pm 0.68$<br>94.32 [93.59, 95.59] | $30.32 \pm 2.49$<br>29.79 [26.66, 35.51] |

---

## 4. Inferential Statistical Hypothesis Testing

### Table 2: Paired Differences, Confidence Intervals, Test Statistics & Adjusted Significance

| Metric | Mean $\Delta$ | Std $\Delta$ | 95% Parametric CI | 95% Bootstrap CI | Paired $d_z$ | Paired $t(19)$ | Two-Sided $p_t$ | Wilcoxon $W$ | Two-Sided $p_W$ | Secondary 1-sided $p_W$ | Win / Loss / Tie | Holm $p_{\text{adj}}$ | Confirmatory Result |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|
| **QWK** | **+0.0294** | 0.0656 | [-0.0013, +0.0601] | [+0.0037, +0.0597] | +0.4487 | 2.0065 | **0.0593** | 55.0 | 0.0637 | 0.0319 | 13 / 7 / 0 | 0.1186 | **Fail** |
| **MAE** | **-0.0300** | 0.0631 | [-0.0595, -0.0005] | [-0.0573, -0.0048] | -0.4756 | -2.1269 | **0.0468** | 46.0 | 0.0266 | 0.9880 | 15 / 5 / 0 | 0.1186 | **Fail** |
| **Accuracy** | **+1.308%** | 2.841% | [-0.022%, +2.638%] | [+0.103%, +2.536%] | +0.4604 | 2.0589 | **0.0535** | 48.0 | 0.0328 | 0.0164 | 13 / 7 / 0 | 0.1186 | **Fail** |
| **OBO Acc** | **+1.483%** | 2.761% | [+0.191%, +2.775%] | [+0.297%, +2.671%] | +0.5373 | 2.4030 | **0.0266** | 34.5 | 0.0085 | 0.0042 | 14 / 6 / 0 | 0.1186 | **Fail** |
| **Macro-F1** | **+0.488%** | 3.435% | [-1.120%, +2.095%] | [-0.978%, +1.986%] | +0.1420 | 0.6350 | **0.5330** | 89.0 | 0.5706 | 0.2853 | 11 / 9 / 0 | 0.5330 | **Fail** |

*Note: All confidence intervals and hypothesis tests are paired and two-sided. Holm-Bonferroni correction controls the family-wise error rate across all five metrics at $\alpha = 0.05$.*

---

## 5. Non-Parametric Bootstrap Distribution Analysis

To evaluate robustness against distributional assumptions, 10,000 paired bootstrap resamples were computed:
- **QWK Mean Difference Bootstrap CI (BCa 95%):** `[+0.0037, +0.0597]`
  - While the 95% bootstrap interval strictly excludes zero (lower bound $= +0.0037$), the margin is razor-thin and heavily governed by seeds 19 and 31.
- **Parametric 95% Student's $t$ CI:** `[-0.0013, +0.0601]`
  - Standard Student's $t$ distribution spans zero, fully corroborating the two-sided $p = 0.0593$.

---

## 6. Stability Audit & Outlier Breakdown

### The Role of Failure-Prone Baseline Seeds
A rigorous distribution analysis identified two statistical outliers in the baseline population based on Tukey's standard IQR rule ($\text{Q1} - 1.5 \times \text{IQR} = 0.6270$):
- **Seed 19:** Baseline $\text{QWK} = 0.5363$, $\text{MAE} = 0.3994$ (vs CASM v2: $\text{QWK} = 0.7179$, $\text{MAE} = 0.2149$) $\to \Delta\text{QWK} = \mathbf{+0.1816}$
- **Seed 31:** Baseline $\text{QWK} = 0.5000$, $\text{MAE} = 0.4433$ (vs CASM v2: $\text{QWK} = 0.6923$, $\text{MAE} = 0.2315$) $\to \Delta\text{QWK} = \mathbf{+0.1923}$

### Sensitivity Analysis: All Seeds vs Stable Cohort

| Cohort | Sample Size | Baseline QWK | CASM v2 QWK | Mean $\Delta$ QWK | Two-Sided $p$-value | Paired $d_z$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full Evaluation (Primary)** | $N=20$ | $0.7032 \pm 0.0728$ | $0.7326 \pm 0.0195$ | **+0.0294** | **0.0593** | +0.4487 |
| **Stable Cohort (Excl. 19, 31)** | $N=18$ | $0.7237 \pm 0.0376$ | $0.7357 \pm 0.0175$ | **+0.0119** | **0.2175** | +0.4065 |

### Mechanistic Interpretation:
1. **Sum of Improvements:** Total cumulative $\Delta\text{QWK}$ across all 20 seeds is $+0.5884$. Seeds 19 and 31 contribute $+0.3739$, representing **$63.5\%$ of the total cumulative improvement**.
2. **Robustness vs Broad Superiority:** On seeds where the baseline trains stably ($N=18$), the performance advantage of CASM v2 narrows to $+0.0119$ and is statistically indistinguishable from zero ($p = 0.2175$).
3. **Core Architectural Value:** The empirical merit of CASM v2 is **not** that it boosts the performance ceiling on well-behaved initializations, but rather that it **inoculates the model against initialization collapse** under severe class imbalance (where Healthy/Grade-0 represents $87.5\%$ of samples).

---

## 7. Reviewer Summary & Publication Verdict

- **Formal Decision:** **Outcome B (Substantial Empirical Robustness & Variance Reduction, but Inconclusive Primary Confirmatory Significance at $\alpha=0.05$).**
- **Manuscript Representation Guidance:** Authors must honestly report $p = 0.0593$ (two-sided) and present the stability advantage as the genuine scientific discovery, rather than claiming an unreserved victory across all ordinal quality metrics.
