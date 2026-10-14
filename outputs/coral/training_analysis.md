# Task 1: CORAL Training Analysis

_Paper: Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading_

---

## 1.1 Final Validation Metrics (Best Checkpoint)

> Training metrics not yet available. Run `train_phase3_coral.py` first.

## 1.2 Training Configuration Summary

        Parameter                                                          Value
         Backbone                         ConvNeXt-Tiny (ImageNet-1K pretrained)
             Head                    CoralHead (shared-w, K-1=2 ordinal outputs)
          Stage 1                             5 epochs, backbone frozen, lr=3e-3
          Stage 2                       up to 20 epochs, full fine-tune, lr=3e-4
        Optimizer                                      AdamW (weight_decay=1e-2)
        Scheduler                                              CosineAnnealingLR
  Mixed Precision                                  AMP FP16 (torch.amp.autocast)
Gradient Clipping                                                            1.0
   Early Stopping                                          patience=5 on val QWK
       Batch Size                                16 (safe for RTX 3050 6GB VRAM)
          Sampler                  WeightedRandomSampler (imbalance ratio 1.84x)
     Augmentation RandomCrop, HFlip, VFlip, Rotation, ColorJitter, RandomErasing

## 1.3 Two-Stage Training Rationale

**Why Two-Stage Training?**

The real_grade domain contains only 1,602 samples across 3 fruit species.
Fine-tuning a 28M-parameter ConvNeXt-Tiny backbone end-to-end from random
head initialization risks catastrophic interference between backbone
representation learning and CORAL head optimization.

**Stage 1 (Head-only, 5 epochs):**
- Backbone frozen at ImageNet weights
- Only the CORAL head (weight w, biases b0, b1) and LayerNorm are trained
- LR = 3e-3 with cosine decay
- Establishes stable ordinal threshold separation before backbone perturbation

**Stage 2 (Full fine-tune):**
- All parameters unfrozen
- LR reduced to 3e-4 (10x lower) with cosine annealing
- Allows domain-specific texture adaptation without destroying pretrained representations
- Early stopping on QWK prevents overfitting on small dataset

**Expected Convergence:**
- Stage 1: QWK increases from ~0.0 to ~0.3-0.5 (head learns ordinal boundaries)
- Stage 2: QWK increases to ~0.5-0.8, then plateaus or early-stops
- Loss: Binary CE typically settles in range [0.3, 0.6] for 3-class ordinal problems
## 1.4 Overfitting Risk Assessment

**Dataset Size vs. Model Capacity:**

| Component | Parameters |
|-----------|------------|
| ConvNeXt-Tiny backbone | ~27.8M |
| LayerNorm (768) | 1,536 |
| CoralHead (w + b) | 770 |
| **Total trainable** | **~27.8M** |

With only 869 training samples, the effective samples-per-parameter ratio
is extremely low (~3.1e-5). Mitigations applied:

1. **Pretrained backbone**: 90%+ of parameters already encode generalizable
   visual features from ImageNet-1K (1.28M images).
2. **Stage 1 freezing**: Prevents 27.8M backbone parameters from overfitting
   during initial head training.
3. **Dropout (p=0.15)**: Applied before the CORAL head.
4. **WeightedRandomSampler**: Ensures minority grade_3 is seen each epoch.
5. **Heavy augmentation**: Effective training set size is multiplied through
   RandomCrop, ColorJitter, and RandomErasing.
6. **Early stopping (patience=5)**: Terminates training before QWK on val degrades.

**Key diagnostic**: Train QWK vs Val QWK gap > 0.15 would indicate overfitting.
