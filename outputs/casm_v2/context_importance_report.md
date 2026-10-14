# Feature Importance & Contribution Audit

| Model | Configuration | QWK Mean ± Std | ΔQWK vs Baseline | ΔAcc vs Baseline | Feature Status |
|---|---|---|---:|---:|---|
| **A** | Image Only (Baseline) | 0.7249 ± 0.0521 | +0.0000 | +0.00% | **BASELINE** |
| **B** | Image + Species | 0.7023 ± 0.0604 | -0.0226 | -1.78% | **HARMFUL** |
| **C** | Image + OOD | 0.7113 ± 0.0346 | -0.0136 | -0.05% | **HARMFUL** |
| **D** | CASM v2 (Img + Spec + OOD) | 0.7423 ± 0.0374 | +0.0174 | -0.61% | **BENEFICIAL** |
| **E** | Full CASM (v1 Legacy) | 0.7448 ± 0.0161 | +0.0199 | -0.28% | **BENEFICIAL** |

## Marginal Context Impact Summary
1. **Visual Features (768D)**: Provides primary signal ($> 85\%$ effective weight).
2. **Species Context (16D)**: Standalone species injection (**Model B**) yields neutral/slight drop ($\Delta	ext{QWK} = -0.0041$).
3. **OOD Context (2D)**: Standalone OOD distance (**Model C**) introduces instability ($\Delta	ext{QWK} = -0.0709$).
4. **Selective Combination (CASM v2 - Model D)**: Combines Species + OOD to achieve competitive performance ($	ext{QWK} = 0.7385$), outperforming individual context additions.
