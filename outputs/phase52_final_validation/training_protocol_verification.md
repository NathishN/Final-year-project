# Phase 5.2 — Training Protocol Verification

## Audit Findings

1. **Precomputed Features:** Confirmed by `phase50_stage5_confirmatory.py`. Training loaded `outputs/features/phase4_dataset.pt` (shape `[56036, 768]`), containing precomputed ConvNeXt-Tiny embeddings.
2. **Frozen Backbone:** The 27.8M-parameter ConvNeXt-Tiny backbone was completely frozen during the 120 confirmatory runs. No raw images were loaded or augmented during confirmatory training.
3. **Trainable Parameters:** Only the projection MLP and MonotonicCORAL ordinal head were optimized (~626,629 parameters for Config A, ~630,730 for Config D).
4. **Optimizer & Loss:** AdamW optimizer with initial learning rate 1e-3, weight decay 1e-4, using the mathematically corrected monotonic ordinal cross-entropy loss.
5. **Batch Size:** 2048 feature vectors per batch.
6. **Implication:** Claims of end-to-end convolutional training are technically inaccurate; the experiments represent frozen-feature ordinal regression and fusion head training.
