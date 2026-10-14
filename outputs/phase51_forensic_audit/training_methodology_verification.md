# Training Methodology Verification Report

## Verified Training Protocol

The confirmatory experiment (`phase50_stage5_confirmatory.py`) uses:

- **Input:** Precomputed ConvNeXt-Tiny feature tensors from `outputs/features/phase4_dataset.pt`
- **Backbone:** FROZEN (no gradient flow through backbone)
- **Trainable parameters:** CORAL head + fusion MLP only (~626K params)
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Epochs:** 15
- **Batch size:** 2048 (feature-level, not image-level)
- **Checkpoint selection:** Validation QWK
- **Optimizer state:** NOT saved (training not reproducible at epoch level)

## Critical Disclosure
The manuscript must be updated to state that:
1. Backbone was frozen; only the CORAL head was trained.
2. Inference speed > 100,000 FPS refers to head-only inference on precomputed features.
3. Full end-to-end throughput includes ConvNeXt-Tiny backbone (~28M params) at ~50-200 FPS.

## Feature Cache Details
- `vis_feats`: shape (56036, 768)
- `species_ids`: shape (56036,)
- `grade_ctx`: shape (56036, 4)
- `ood_ctx`: shape (56036, 2)
- `unc_ctx`: shape (56036, 2)
- `targets`: shape (56036,)
- `splits`: shape (56036,)
- `domains`: shape (56036,)
- `fruits`: shape (56036,)
- `paths`: shape (56036,)