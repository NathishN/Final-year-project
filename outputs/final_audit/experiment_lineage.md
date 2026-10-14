# Research Experiment Lineage & Traceability Map

This document traces the complete lineage of experiments, datasets, models, and metric claims across Phases 1 through 5.0.

```mermaid
graph TD
    P1[Phase 1: Species Classification] --> CKPT1[outputs/checkpoints/best_model.pth]
    CKPT1 --> FEAT[outputs/features/phase4_dataset.pt (ConvNeXt 768-D)]
    P2[Phase 2: Open-Set Mahalanobis] --> OOD1[Synthetic Gaussian Artifact (Audit Defect)]
    P3[Phase 3: CORAL Ordinal Head] --> CORAL1[Unconstrained Thresholds (Audit Defect)]
    P4[Phase 4: CASM Multi-Modal Ablation] --> AUDIT45[Phase 4.5 Audit: Outcome D]
    AUDIT45 --> SPLIT2[split_v2: Session 20200730 -> Test (Grade 2 n=119)]
    AUDIT45 --> MHEAD[MonotonicCORALHead: Cumulative Softplus Cutpoints]
    AUDIT45 --> ROBSPEC[RobustSpeciesEmbedding: Unknown Fallback Index 5]
    AUDIT45 --> REALOOD[Real-Image OOD: 5,000 Images from open-set-dataset]
    SPLIT2 & MHEAD & ROBSPEC & REALOOD --> P5[Phase 5.0: 30-Seed Confirmatory Study]
    P5 --> VERDICT[Phase 5.0 Outcome C: Context Conditioning Degrades QWK]
```

## Detailed Evolution by Phase

### Phase 1: ConvNeXt-Tiny Species Backbone
- **Dataset:** 5 fruit species (Apple, Banana, Mango, Orange, Tomato).
- **Backbone:** ConvNeXt-Tiny (ImageNet-1k pretrained).
- **Checkpoint:** `outputs/checkpoints/best_model.pth`.
- **Status:** Validated, foundational visual feature extractor.

### Phase 2: Open-Set Mahalanobis Formulation
- **Defect Identified:** Evaluated on synthetic Gaussian noise rather than actual images, producing an artificial 100% AUROC.
- **Resolution in Phase 5.0:** Extracted 5,000 real images across Cherry, Kiwi, Peach, Pineapple, and Plum, achieving true 84.22% AUROC.

### Phase 3: CORAL Ordinal Regression
- **Defect Identified:** Unconstrained threshold parameterization allowed $b_k$ inversions, causing negative discrete probabilities.
- **Resolution in Phase 5.0:** Monotonic parameterization via cumulative softplus decrements mathematically guarantees non-negative probabilities.

### Phase 4: Context-Aware Severity Modeling (CASM v1)
- **Defect Identified:** Circular grade context, missing Grade 2 test samples, extreme 175:1 inverse loss weights.
- **Resolution in Phase 5.0:** Formulated CASM v2 without circular features, evaluated on split_v2 with Effective Number weighting.

### Phase 5.0: Confirmatory Empirical Validation
- **Execution:** 4 Configurations × 30 Prespecified Seeds ($N=120$ runs).
- **Empirical Finding:** Configuration A (Image Only) significantly outperforms Configuration B (Image + Species) on QWK ($\Delta = -0.0360, p = 0.000259$). Context conditioning hypothesis disproven.