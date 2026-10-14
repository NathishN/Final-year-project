# Context Significance Report

Paired t-tests vs Image-Only (Model A) across 5 seeds.

| Model | Config | Acc Δ | MAE Δ | QWK Δ | F1 Δ | QWK p | Sig | Direction |
|-------|--------|-------|-------|-------|------|-------|-----|-----------|
| B | Image + Species | -1.2050 | +0.0266 | -0.0041 | -2.6813 | 0.5790 | NO | WORSE |
| C | Image + OOD | -0.8840 | +0.0246 | -0.0709 | +2.9000 | 0.1581 | NO | WORSE |
| D | Image + Grade | -1.5353 | +0.0256 | -0.0359 | +0.8811 | 0.5341 | NO | WORSE |
| E | Image + Uncertainty | -0.6795 | +0.0377 | -0.0925 | -0.0863 | 0.3725 | NO | WORSE |
| F | Image + Species + OOD | +0.4515 | -0.0120 | +0.0163 | +2.1996 | 0.3025 | NO | BETTER |
| G | Image + Species + Grade | -0.7503 | +0.0135 | +0.0092 | -2.6085 | 0.7095 | NO | BETTER |
| H | Image + OOD + Grade | -0.4153 | -0.0085 | -0.0116 | +5.0424 | 0.5972 | NO | WORSE |
| I | Full CASM | +0.8621 | -0.0201 | +0.0183 | +2.9318 | 0.4445 | NO | BETTER |

## Summary
- Statistically BETTER (p<0.05): None
- Statistically WORSE  (p<0.05): None