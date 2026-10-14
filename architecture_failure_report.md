# Architecture Failure Analysis Report

| Metric | Model A (Image Only) | Model I (Full CASM) | Change |
|--------|---------------------|---------------------|--------|
| QWK | 0.7222±0.0210 | 0.7405±0.0370 | +0.0183 |
| Acc | 87.06%±0.65% | 87.93%±0.80% | +0.86% |
| MAE | 0.2276±0.0061 | 0.2075±0.0321 | -0.0201 |

## Failure Hypotheses
| # | Hypothesis | Verdict | Severity |
|---|-----------|---------|----------|
| H1 | Over-Parameterization | CONTRIBUTING | Medium |
| H2 | Circular Feature (Grade←VisFeats) | CONFIRMED | High |
| H3 | Redundant Features (OOD-Prob≈1-MaxConf) | CONFIRMED | High |
| H4 | OOD Calibration Mismatch (species vs severity) | CONTRIBUTING | Medium |
| H5 | FiLM Amplifies Low-Signal Context | CONFIRMED | High |
| H6 | High Seed Variance in Context Models C,D,E | OBSERVED | Medium |

Low-signal features: OOD P(OOD), Unc Entropy, Unc MaxConf

## Recommendations
1. REMOVE Grade context (circular dependency)
2. REMOVE Unc-MaxConf (redundant with OOD-Prob)
3. RETAIN Species embedding (independent biological prior)
4. RETAIN OOD Mahalanobis (Model F QWK=0.7385 > Model B QWK=0.7181)
5. REPLACE FiLM with simple concatenation fusion