# Context Significance Report (Phase 4.2)

Paired t-tests vs Image-Only baseline (Model A) across 5 seeds with Holm-Bonferroni correction.

| Model | Configuration | Acc Δ | MAE Δ | QWK Δ | F1 Δ | Raw p | Holm-Adj p | Significance Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **B** | Image + Species | -1.7776 | +0.0477 | -0.0226 | -3.7288 | 0.6262 | 1.0000 | **WORSE (NO (Not Sig))** |
| **C** | Image + OOD | -0.0503 | +0.0015 | -0.0136 | +1.2494 | 0.7007 | 0.7007 | **WORSE (NO (Not Sig))** |
| **D** | CASM v2 (Img + Spec + OOD) | -0.6103 | -0.0003 | +0.0174 | -0.0969 | 0.5525 | 1.0000 | **BETTER (NO (Not Sig))** |
| **E** | Full CASM (v1 Legacy) | -0.2769 | -0.0001 | +0.0199 | -0.6702 | 0.4103 | 1.0000 | **BETTER (NO (Not Sig))** |

## Key Inference & Limitations
- **Sample Size Constraint:** 5 seeds provide $N=5$ degrees of freedom ($df=4$), yielding low statistical power for subtle $\Delta	ext{QWK} < 0.02$ variations.
- **Conclusion:** Neither isolated context streams nor CASM v2 establish statistically significant QWK superiority over the Image-Only baseline under rigorous multiplicity correction.
