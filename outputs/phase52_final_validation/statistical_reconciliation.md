# Phase 5.2 — Statistical Reconciliation & Multiplicity Analysis

## 1. Resolution of Wilcoxon Discrepancy

Historical documents previously cited competing Wilcoxon statistics:
- Phase 4.4/4.5: $W = 59$
- Phase 5.0 Report: $W = 62$ ($p = 0.000928$)
- Phase 5.1 Recomputation: $W = 72$ ($p = 0.000555$)

### Root-Cause Analysis:
The 30 paired QWK differences between Configuration B and Configuration A contain **zero ties with zero** (all 30 seeds have non-zero differences). Executing `scipy.stats.wilcoxon(diff_B_minus_A, zero_method='wilcox')` yields precisely **$W = 72.0$**, with asymptotic/exact $p = 0.000555$. The sum of negative ranks is $393.0$ and positive ranks is $72.0$ (total rank sum $30 \times 31 / 2 = 465$). SciPy returns the smaller rank sum, which is **72.0**. Values of 59 and 62 in earlier reports were artifacts of manual transcription or uncorrected subset calculations. Both parametric ($t(29) = -4.159, p = 0.000260$) and non-parametric ($W = 72, p = 0.000555$) tests decisively reject the null hypothesis of equal performance, demonstrating a statistically significant performance drop when species conditioning is added.

## 2. Paired Contrasts with Multiplicity Policy

| Contrast | Mean Diff | 95% CI | Cohen's $d_z$ | Paired $t(29)$ | Raw $p$ | Holm-Bonferroni $p$ | Wilcoxon $W$ | Wilcoxon $p$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B vs A (Species Conditioning) | -0.0360 | [-0.0537, -0.0183] | -0.7593 | -4.1586 | 0.000260 | 0.000260 | 72.0 | 0.000555 |
| C vs A (OOD Context) | -0.0700 | [-0.0902, -0.0498] | -1.2947 | -7.0914 | 0.000000 | 0.000000 | 16.0 | 0.000000 |
| D vs A (Species + OOD) | -0.0848 | [-0.1008, -0.0687] | -1.9704 | -10.7923 | 0.000000 | 0.000000 | 0.0 | 0.000000 |

## 3. Scientific Implication

Even after stringent Holm-Bonferroni family-wise error rate control across all three pairwise contrasts, all degradations remain highly significant ($p_{adj} < 0.001$). Configuration A (Image-Only) strictly and robustly outperforms all context-conditioned variants on the evaluation benchmark.
