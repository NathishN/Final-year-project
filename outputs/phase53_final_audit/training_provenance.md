# Forensic Training & Checkpoint Provenance Audit

**Audit Timestamp:** 2026-10-14 10:21:54  

**Backbone Architecture:** ConvNeXt-Tiny (27,820,148 parameters) — **FROZEN**  

**Feature Cache Source:** `outputs/features/phase4_dataset.pt` (768-D)  

---

## 1. Checkpoint Inventory & Integrity Summary

| Repository Cohort | Checkpoints Path | Expected | Verified on Disk | Avg Checkpoint Size | State Dict Keys |
|---|---|---:|---:|---:|---:|
| **Historical Confirmatory (Split V2)** | `outputs/casm_v2/phase50_checkpoints` | 120 | **120** | ~2.51 MB | 19 keys |
| **Session-Held-Out Retrained** | `outputs/phase52_final_validation/session_holdout_retrain/checkpoints` | 120 | **120** | ~2.51 MB | 19 keys |


## 2. Model Architecture & Trainable Parameter Breakdown

The 120 confirmatory runs and 120 retrained holdout runs did **not** train end-to-end convolutional weights. The ConvNeXt-Tiny feature extractor remained frozen:

- **Backbone Parameters:** 27.8M (Frozen, pre-extracted to `outputs/features/phase4_dataset.pt`).

- **Trainable Head Parameters:** ~626,180 parameters (Projection MLP: 768 -> 512 -> 256; MonotonicCORAL ordinal linear head with softplus strictly positive threshold cutoffs).

- **Per-Configuration Head Capacity:**

  - Config A (Image Only): 626,629 parameters

  - Config B (Image + Species): 627,653 parameters

  - Config C (Image + OOD): 629,705 parameters

  - Config D (Image + Species + OOD): 630,729 parameters


## 3. Optimization Hyperparameters & Protocol

- **Optimizer:** AdamW

- **Base Learning Rate:** 0.001 (Cosine annealing)

- **Weight Decay:** 0.0001

- **Batch Size:** 2048 (feature vectors)

- **Epochs:** 15

- **Loss Function:** Monotonic CORAL ordinal binary cross-entropy with class re-weighting (Effective Number of Samples $\beta=0.999$, weights: `[0.4433, 0.7762, 2.3594, 0.5545, 0.8667]`)

- **Model Selection Criterion:** Lowest validation set MAE (evaluated strictly on validation partition without test set leakage).


## 4. Checkpoint Serialization & Log Provenance Disclosures

1. **Absence of Optimizer States:** Checkpoint `.pth` files serialize strictly the PyTorch model `state_dict` (parameters and buffers). They do not serialize AdamW momentum buffers or optimizer states.

2. **Absence of Epoch-by-Epoch Checkpoints:** Only the single best checkpoint per configuration per seed is saved (`model_{cfg}_seed{seed}_best.pth`). Intermediate epoch snapshots were discarded to conserve storage.

3. **Complete Prediction Provenance:** Prediction tensors (`preds_{cfg}_seed{seed}.pt`) for all 120 historical runs and 120 session holdout runs are fully preserved and match checkpoint weights deterministically.

