# Task 5: Statistical Significance Testing

---

## 5.1 Multi-Seed Results Table

> Multi-seed results not yet available.

## 5.2 Paired t-test Methodology

**Null Hypothesis (H0):**
No significant difference in mean QWK between CORAL and Cross-Entropy across 5 seeds.

**Alternative Hypothesis (H1):**
CORAL achieves significantly higher QWK than Cross-Entropy.

**Test selected:** One-sided paired t-test (parametric, alpha = 0.05).

**Justification for paired test:**
- Each seed produces matched observations (same data, same random split).
- Pairing removes between-seed variability as a confound.
- 5 pairs satisfies minimum degrees of freedom for t-test (df=4).

**Significance threshold:** p < 0.05 (one-sided). Bonferroni correction for 3
simultaneous metrics: alpha_corrected = 0.05/3 = 0.017.

**Effect size (Cohen's d):**

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.2     | Negligible     |
| 0.2-0.5   | Small          |
| 0.5-0.8   | Medium         |
| > 0.8     | Large          |

## 5.3 t-test Computation

> Awaiting multi-seed results.

Will compare CORAL vs CE across seeds [42, 123, 2025, 999, 777] for QWK, MAE, Accuracy.

## 5.4 Statistical Recommendations for IEEE Submission

1. **Report mean +/- std** for all metrics across 5 seeds.
2. **Use paired t-test** (n=5) -- paired removes seed-variance confound.
3. **Report Cohen's d** alongside p-value -- p-value alone is insufficient.
4. **Bonferroni correction** for 3 simultaneous metrics: alpha = 0.017.
5. **McNemar's test** for per-sample accuracy comparison.
6. **95% CI** on mean QWK improvement: CORAL_mean +/- 1.96 * SE.

IEEE Reviewer expectation:
> "Authors must demonstrate statistical significance of their proposed method
> over baselines. Single-seed results are insufficient for publication in IEEE
> Transactions journals."
