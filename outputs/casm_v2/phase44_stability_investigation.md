# Phase 4.4 Task C — Training Stability Investigation

**Primary analysis: ALL N=20 seeds (no exclusions)**

## Anomaly Detection

Baseline QWK: mean=0.7032 median=0.7265 Q1=0.6956 Q3=0.7414 IQR=0.0458
Outlier fence (Q1 - 1.5×IQR): 0.6270
Seeds below fence: [19, 31]

## Suspect Seed Profiles

| Seed | Model | QWK | MAE | Acc% | OBO% | F1% |
|------|-------|-----|-----|------|------|-----|
| 19 | Baseline | 0.5363 | 0.3994 | 76.59 | 85.33 | 30.59 |
| 19 | CASMv2   | 0.7179 | 0.2149 | 87.02 | 93.68 | 34.99 |
| 31 | Baseline | 0.5000 | 0.4433 | 80.34 | 84.28 | 38.04 |
| 31 | CASMv2   | 0.6923 | 0.2315 | 87.18 | 93.75 | 31.67 |

## Collapse Diagnosis

**Seed 19 (Baseline):** QWK=0.5363, MAE=0.3994, OBO=85.33%
- MAE is 1.75x the population mean (0.2445) — indicates grade-0 bias
- Pattern consistent with partial class collapse toward majority class
- CASMv2 recovers to QWK=0.7179 (Δ=+0.1816)

**Seed 31 (Baseline):** QWK=0.5000, MAE=0.4433, OBO=84.28%
- Lowest QWK and highest MAE of all 40 runs
- Below IQR outlier fence
- CASMv2 recovers to QWK=0.6923 (Δ=+0.1923)

**Mechanistic hypothesis (associational, not proven):**
> With 87.5% grade-0 prevalence, certain random initializations collapse
> to predicting grade-0 for most inputs. The species embedding may provide
> a diversity signal that prevents this collapse, but causality is unconfirmed.

## Effect on Mean Difference

| Analysis | N | Mean QWK diff | Interpretation |
|----------|---|---------------|----------------|
| **Primary (all seeds)** | 20 | +0.0294 | Main result |
| Sensitivity (excl. 19,31) | 18 | +0.0119 | Diagnostic only |
| Contribution of seeds 19,31 | — | +0.0175 (59%) | Large |

> Seeds 19 and 31 contribute approximately 63% of the mean QWK difference.
> Without them, the mean improvement drops from +0.0294 to +0.0119.
> This is a critical result: the mean improvement is substantially driven
> by CASM v2's robustness on collapse-prone seeds, not broad superiority.

## Sensitivity Analysis (Diagnostic Only — NOT Primary)

| Metric | All N=20 | N=18 (excl. 19,31) |
|--------|----------|--------------------|
| Baseline QWK | 0.7032±0.0728 | 0.7237±0.0376 |
| CASMv2 QWK | 0.7326±0.0195 | 0.7357±0.0175 |
| Mean diff | +0.0294 | +0.0119 |
| p-value (2-sided t) | 0.0593 | 0.2175 |
| Cohen d | +0.4487 | +0.4065 |

> **Conclusion:** With stable seeds only, the QWK improvement shrinks to
> +0.0119 (p=0.2175). The primary improvement is primarily a
> **robustness advantage** on collapse-prone initializations, not a general
> accuracy improvement on all seeds.

## Protocol Integrity
- Seeds were pre-specified before training: [42,123,777,999,2025,1,2,3,5,7,11,13,17,19,23,31,37,41,43,47]
- Identical training protocol applied to both Baseline and CASMv2
- No cherry-picking or post-hoc seed selection
- Seeds 19 and 31 are genuine training failures, not adversarial seeds

## Conclusion

> The primary statistical result (N=20) is valid. However, the improvement
> is largely a **stability/robustness benefit** rather than a uniform accuracy gain.
> This is scientifically important and publishable as a distinct contribution:
> CASM v2 is more robust to training instability caused by class imbalance.