# Feature Importance Report

| Context | Model | ΔQWK | ΔAcc | Verdict |
|---------|-------|------|------|---------|
| Species | B | -0.0041 | -1.2049% | MARGINAL |
| OOD | C | -0.0709 | -0.8840% | HARMFUL |
| Grade | D | -0.0359 | -1.5353% | HARMFUL |
| Uncertainty | E | -0.0926 | -0.6795% | HARMFUL |

## Full Config Ranking (by QWK Mean)
| Rank | Model | Config | QWK Mean±Std | ΔQWK |
|------|-------|--------|--------------|------|
| 1 | I | Full CASM | 0.7405±0.0370 | +0.0183 |
| 2 | F | Image + Species + OOD | 0.7385±0.0200 | +0.0163 |
| 3 | G | Image + Species + Grade | 0.7313±0.0402 | +0.0091 |
| 4 | A | Image Only | 0.7222±0.0210 | +0.0000 |
| 5 | B | Image + Species | 0.7181±0.0188 | -0.0041 |
| 6 | H | Image + OOD + Grade | 0.7106±0.0327 | -0.0116 |
| 7 | D | Image + Grade | 0.6863±0.1016 | -0.0359 |
| 8 | C | Image + OOD | 0.6513±0.0963 | -0.0709 |
| 9 | E | Image + Uncertainty | 0.6296±0.1912 | -0.0926 |