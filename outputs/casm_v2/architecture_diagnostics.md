# Architecture Diagnostics & Collinearity Report

## 1. Model Parameter Count Comparison

| Model ID | Configuration | Parameter Count | Context Dim | Fusion Mechanism |
|---|---|---:|---:|---|
| **Model A** | Image Only | 426,884 | 0D | None (Linear Proj) |
| **Model B** | Image + Species | 431,092 | 16D | Concatenation |
| **Model C** | Image + OOD | 427,396 | 2D | Concatenation |
| **Model D** | CASM v2 (Img + Spec + OOD) | 431,604 | 18D | Concatenation |
| **Model E** | Full CASM (v1 Legacy) | 563,940 | 24D | FiLM Modulation & Gating |

## 2. Structural Diagnostic Findings
1. **Parameter Efficiency**: CASM v2 requires **23.5% fewer parameters** than Full CASM v1 (431k vs 564k), mitigating over-parameterization risks.
2. **FiLM Gating Removal**: Removing FiLM scale/shift parameters prevents low-variance context features from modulating ConvNeXt visual embeddings.
3. **Collinearity Prevention**: Eliminating Grade context ($P(G_1), P(G_2), P(G_3)$) removes $R = -0.9847$ collinearity and breaks the circular information loop back to visual features.
4. **Softmax Redundancy Removal**: Eliminating Entropy and MaxConf removes $R = -1.0000$ mathematical equivalence with $1 - P(\text{OOD})$.
