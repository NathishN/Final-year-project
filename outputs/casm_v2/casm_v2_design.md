# CASM v2 Architecture Design Specification

## Executive Overview
**CASM v2 (Species & OOD Selective Fusion Network)** addresses the failure modes of CASM v1 by removing circular feature dependencies, redundant uncertainty signals, and noisy FiLM gating modulations.

## Architecture Block Diagram
```
Input: Image X, Species Index s (0..4, fallback 5), Mahalanobis OOD Vector d_mah [2D]
 ├── Visual Stream:  ConvNeXtTiny(X) [768D] ──► Linear(768->512) ──► LN ──► GELU ──► Linear(512->256) ──► LN ──► GELU [256D]
 ├── Species Stream: Embedding(6, 16)[s]                                                                   [16D]
 └── OOD Stream:     Mahalanobis Distance Metric [d_mah, p_ood]                                            [2D]
                                                                                                           
 Selective Fusion:  Concatenation([f_vis, f_spec, f_ood])                                                  [274D]
 Fusion MLP:         Linear(274->256) ──► LN ──► GELU ──► Dropout(0.2) ──► Linear(256->128) ──► LN ──► GELU ──► Dropout(0.1) [128D]
 Ordinal Head:       CORALHead(128, K=5) ──► 4 Binary Task Logits ──► Count(sigmoid(logits) > 0.5)           Rank in {0,1,2,3,4}
```

## Component Rationale
1. **Visual Stream (256D)**: Preserves rich ConvNeXt visual representation of localized surface blemishes.
2. **Species Embedding (16D)**: Encodes biological ripening priors without hand-crafted features. Handles unknown species via index 5.
3. **OOD Metric (2D)**: Provides distance-to-centroid score strictly fitted on training features.
4. **Direct Concatenation**: Replaces FiLM modulation to prevent low-signal context vectors from distorting valid visual features.
5. **CORAL Ordinal Head**: Enforces rank consistency across ordinal thresholds via shared weight vector W.
