# Phase 4.4 Task A & B — Metric Formulation, Effect Size & Multiple Testing Audit

**Auditor:** Statistical ML Specialist & IEEE Transactions Senior Reviewer  
**Date:** October 2026  
**Repository:** `C:\fruit_dataset`  
**Artifacts Audited:** `outputs/casm_v2/phase44_recalculated_metrics.csv`, `outputs/casm_v2/phase44_statistical_report_data.csv`

---

## 1. Executive Summary

This metric audit investigates discrepancies between preliminary reports and rigorous statistical recalculations. It addresses three major methodological challenges:
1. **Cohen's $d$ Discrepancy:** Prior documents reported Cohen's $d = +0.5520$ based on independent between-model pooled standard deviation ($s_{\text{pooled}}$). The scientifically correct metric for paired repeated-seed designs is paired Cohen's $d_z = \bar{D} / s_D = \mathbf{+0.4487}$.
2. **Ghost Class Macro-F1 Deflation:** In the test set ($N=12,714$), Grade 2 ("Moderate Defect") has zero ground-truth instances. Evaluating a standard 5-class unweighted Macro-F1 enforces an automatic $0.0$ score for Class 2, artificially deflating overall Macro-F1 to $\sim 30\%$.
3. **Multiple Testing Failure under Holm-Bonferroni:** While unadjusted two-sided $p$-values for MAE ($p=0.0468$) and OBO Accuracy ($p=0.0266$) fell below $\alpha = 0.05$, **no metric survives family-wise error rate control** (all adjusted $p \ge 0.1186$).

---

## 2. Effect Size Formulation & Discrepancy Audit

### The Two Formulations of Cohen's $d$

In experimental ML benchmarking, confusing independent-samples Cohen's $d$ with paired Cohen's $d_z$ is a frequent source of inflated or distorted effect size claims.

#### 1. Independent Pooled Formulation ($d_s$):
$$d_s = \frac{\bar{X}_{\text{CASM}} - \bar{X}_{\text{Baseline}}}{\sqrt{\frac{(n-1)s_{\text{CASM}}^2 + (n-1)s_{\text{Baseline}}^2}{2n - 2}}}$$
- Values: $\bar{X}_1 = 0.7326$, $s_1 = 0.0195$, $\bar{X}_0 = 0.7032$, $s_0 = 0.0728$, $n = 20$.
- Pooled Standard Deviation: $s_{\text{pooled}} = \sqrt{(0.0195^2 + 0.0728^2)/2} = 0.05327$.
- Yields: $d_s = 0.02942 / 0.05327 = \mathbf{+0.5523}$ (reported in preliminary Phase 4.3 notes as $+0.5520$).

#### 2. Paired Difference Formulation ($d_z$ — The Methodologically Correct Metric):
In a matched-pairs design where both models are evaluated on identical test sets across identical random seeds, the relevant random variable is the paired difference $D_i = X_{\text{CASM}, i} - X_{\text{Baseline}, i}$.
$$d_z = \frac{\bar{D}}{s_D} = \frac{\mu_D}{\sigma_D}$$
- Sample Mean Difference: $\bar{D} = +0.02942$
- Sample Standard Deviation of Differences: $s_D = 0.065573$
- Paired Effect Size: $d_z = \frac{0.02942}{0.065573} = \mathbf{+0.4487}$

### Audit Assessment
$d_s$ ignores the positive covariance between paired seeds ($r = 0.441$), which in this instance causes an overestimation of the standardized effect size by $+23.1\%$ relative to $d_z$ ($0.5520$ vs $0.4487$). All IEEE Transactions publication submissions must report $d_z = +0.4487$ explicitly labeled as paired Cohen's $d$.

---

## 3. Ghost Class Analysis: The Grade 2 Anomaly in Test Split

### Mathematical Derivation of Macro-F1 Deflation
The standard multi-class Macro-F1 is defined as:
$$\text{Macro-F1} = \frac{1}{K} \sum_{k=0}^{K-1} F_{1, k}$$
where $K=5$ corresponding to grades $\{0, 1, 2, 3, 4\}$.

In the test partition ($N=12,714$):
- Ground truth count for Grade 2: $N_2 = 0$.
- If a model predicts zero samples as Grade 2: Precision is undefined ($0/0$), Recall is undefined ($0/0$). Under Scikit-learn's default convention (`zero_division=0`), $F_{1, 2} = 0.0$.
- If a model predicts $M > 0$ samples as Grade 2: All $M$ predictions are False Positives. Precision $= 0/M = 0$, Recall $= 0/0 = 0 \implies F_{1, 2} = 0.0$.

Consequently, for any model evaluated on the test set:
$$F_{1, 2} \equiv 0.0000$$
$$\text{Macro-F1}_{5\text{-class}} = \frac{F_{1, 0} + F_{1, 1} + 0.0 + F_{1, 3} + F_{1, 4}}{5} = 0.80 \times \text{Macro-F1}_{4\text{-class}}$$

### Empirical Demonstration
| Model | 5-Class Macro-F1 | 4-Class Macro-F1 (Active Classes) | True Multi-class Capability |
|---|---:|---:|---|
| Baseline | 29.83% | **37.29%** | Evaluates active classes 0, 1, 3, 4 |
| CASM v2 | 30.32% | **37.90%** | Evaluates active classes 0, 1, 3, 4 |

**Audit Recommendation:** In the manuscript, authors must clarify that Grade 2 was withheld or absent from the test split, and report both the 5-class Macro-F1 (for protocol consistency) and the active 4-class Macro-F1 to explain why values appear modest ($\sim 30\%$).

---

## 4. Hypothesis Testing: Two-Sided vs One-Sided Rigor

Preliminary informal drafts suggested reporting one-sided $p$-values ($p=0.0297$ for paired $t$-test, $p=0.0319$ for Wilcoxon) to claim statistical significance at $\alpha = 0.05$.

### Scientific Reviewer Rebuttal:
1. **Directionality cannot be assumed a priori:** Adding context features and auxiliary MLP layers introduces parameter overhead, optimization complexity, and potential class-prior overfitting. If context embeddings were misaligned, CASM v2 could degrade performance (as observed in Config C, where QWK degraded to $0.6882$).
2. **IEEE Transactions Standard:** Confirmatory hypothesis testing in peer-reviewed machine learning venues mandates **two-sided tests** unless an explicit pre-registered directional equivalence protocol exists.
3. **Primary Test Statistic:**
   - Paired Student's $t$-test (two-sided): $t(19) = 2.0065, p = \mathbf{0.0593}$
   - Wilcoxon signed-rank test (two-sided): $W = 55.0, p = \mathbf{0.0637}$

Both primary tests consistently fail to reject the null hypothesis at the conventional threshold of $\alpha = 0.05$. Reporting one-sided tests as the primary confirmatory claim would constitute outcome-driven reporting ("$p$-hacking").

---

## 5. Multiple Testing Correction (Holm-Bonferroni)

When simultaneously evaluating 5 performance metrics as confirmatory evidence of superiority, the family-wise error rate ($\text{FWER}$) must be controlled.

### Holm-Bonferroni Procedure ($\alpha = 0.05$, $m = 5$ tests)
Metrics ranked in ascending order of raw two-sided $p$-value:

| Rank $j$ | Metric | Raw Two-Sided $p_j$ | Target Threshold $\frac{\alpha}{m - j + 1}$ | Holm-Adjusted $p_{\text{adj}}$ | Significant at $\alpha = 0.05$? |
|---|---|---:|---:|---:|:---:|
| 1 | OBO Accuracy | **0.0266** | $0.05 / 5 = 0.0100$ | $0.1186$ | **FAIL** |
| 2 | MAE | **0.0468** | $0.05 / 4 = 0.0125$ | $0.1186$ | **FAIL** |
| 3 | Accuracy | **0.0535** | $0.05 / 3 = 0.0167$ | $0.1186$ | **FAIL** |
| 4 | QWK | **0.0593** | $0.05 / 2 = 0.0250$ | $0.1186$ | **FAIL** |
| 5 | Macro-F1 | **0.5330** | $0.05 / 1 = 0.0500$ | $0.5330$ | **FAIL** |

### Audit Finding
Under family-wise error rate control, **none of the 5 metrics achieve statistical significance at $\alpha = 0.05$**. Claims that CASM v2 provides statistically validated improvements across ordinal quality grading metrics cannot be sustained without reservation.

---

## 6. Audit Verdict

1. **Cohen's $d$:** Corrected to paired $d_z = +0.4487$.
2. **Significance Verdict:** Misses confirmatory significance ($p = 0.0593$ two-sided; $p_{\text{adj}} = 0.1186$ Holm-Bonferroni).
3. **Macro-F1 Interpretation:** Caveat regarding test set Grade 2 support must be prominently documented.
