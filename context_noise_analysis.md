# Context Noise Analysis Report

| Feature | Mutual Info | Spearman R | p-value | Signal |
|---------|-------------|------------|---------|--------|
| Species ID | 0.027618 | +0.1258 | 0.0000 | **MEDIUM** |
| Grade P(G1) | 0.081413 | +0.2183 | 0.0000 | **MEDIUM** |
| Grade P(G2) | 0.084385 | +0.3242 | 0.0000 | **HIGH** |
| Grade P(G3) | 0.081408 | -0.2183 | 0.0000 | **MEDIUM** |
| Grade MaxConf | 0.084568 | -0.3242 | 0.0000 | **HIGH** |
| OOD Mah Dist | 0.129532 | -0.3546 | 0.0000 | **HIGH** |
| OOD P(OOD) | 0.027734 | +0.0443 | 0.0000 | **LOW/NOISE** |
| Unc Entropy | 0.027622 | +0.0597 | 0.0000 | **LOW/NOISE** |
| Unc MaxConf | 0.027864 | -0.0443 | 0.0000 | **LOW/NOISE** |

## Redundant Feature Pairs (|R|>0.70)
| A | B | Pearson R |
|---|---|-----------|
| Grade-P1 | Grade-P3 | -0.9847 |
| Grade-P2 | Grade-MaxConf | -0.9808 |
| OOD-Prob | Unc-Ent | +0.9518 |
| OOD-Prob | Unc-MaxConf | -1.0000 |
| Unc-Ent | Unc-MaxConf | -0.9518 |

## Root Cause Summary
1. Grade context derived from same visual backbone → circular dependency
2. OOD-Prob ≈ 1 − Unc-MaxConf → mathematically redundant
3. FiLM gating amplifies low-signal features instead of suppressing them
4. OOD Mahalanobis calibrated for species, not severity ordering