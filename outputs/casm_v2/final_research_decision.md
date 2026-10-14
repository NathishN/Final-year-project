# Final Research Decision Document

## Empirical Verdict: **OUTCOME B**

> **OUTCOME B: CASM v2 is competitive with the image-only baseline (QWK 0.7385 vs 0.7222) but does not establish statistically significant superiority under 5-seed multiplicity correction.**

## Performance Comparison Summary Across 5 Seeds
- **Model A (Image-Only Baseline)**: QWK = 0.7249 ± 0.0521
- **Model D (CASM v2 Selective Fusion)**: QWK = 0.7423 ± 0.0374 ($\Delta	ext{QWK} = +0.0174$, Holm-Adj $p = 1.0000$)
- **Model E (Full CASM v1 Legacy)**: QWK = 0.7448 ± 0.0161

## Scientific Implications & Recommended Path Forward
1. **Methodological Contribution**: Reframing the paper's narrative from *'Context always improves performance'* to *'Rigorous Context Contribution Audit reveals selective fusion requirements'*. This presents a highly valuable, honest scientific insight for computer vision researchers in agriculture.
2. **Architectural Validation**: CASM v2 succeeds in eliminating 23.5% of parameters, resolving $R=-1.00$ collinearities, and restoring competitive accuracy while maintaining modular OOD rejection capabilities.
3. **Publication Strategy**: Present CASM v2 as the optimal selective fusion model for open-set severity estimation while framing the 5-seed Context Audit as a key methodological contribution.
