# Phase 5.6 — Architecture, Training Protocol & Inference Benchmark Audit

**Audit Timestamp:** 2026-10-14 13:47:26  

**Audit Scope:** Verification of backbone, head architecture, exact parameter census, training hyperparameters, loss formulation, monotonic cutpoints, and empirical hardware latency benchmarks.

---

## 1. Complete Model Architecture & Parameter Census


The system uses a two-tier architecture: a frozen pre-trained ConvNeXt-Tiny vision foundation model coupled to a lightweight, structurally monotonic ordinal classification head:


### Tier 1: Vision Feature Backbone (Frozen)

- **Architecture:** ConvNeXt-Tiny (Liu et al., 2022)

- **Pretrained Weights:** ImageNet-1k default checkpoint (`outputs/checkpoints/best_model.pth`)

- **Input Resolution:** $224 \times 224 \times 3$ RGB

- **Normalization:** ImageNet mean $\mu = [0.485, 0.456, 0.406]$, std $\sigma = [0.229, 0.224, 0.225]$

- **Output Feature Dimension:** 768-D global average pooled embedding

- **Backbone Parameter Count:** **27,820,128 parameters** (Frozen during training)


### Tier 2: Trainable Multi-Stage Ordinal Head (Config A)

The trainable head consists of visual projection, feature fusion, and the monotonic ordinal cutpoint layer. Forensic layer-by-layer accounting from `outputs/casm_v2/phase50_checkpoints/model_A_seed83_best.pth` reveals:


| Module / Layer | Operation / Dimensions | Weights Count | Biases Count | Total Layer Parameters |

|:---|:---|---:|---:|---:|

| `vis_proj.0` | Linear(768 $\to$ 512) | 393,216 | 512 | 393,728 |

| `vis_proj.1` | LayerNorm(512) | 512 | 512 | 1,024 |

| `vis_proj.2,3` | ReLU + Dropout(0.2) | 0 | 0 | 0 |

| `vis_proj.4` | Linear(512 $\to$ 256) | 131,072 | 256 | 131,328 |

| `vis_proj.5` | LayerNorm(256) | 256 | 256 | 512 |

| `fusion_mlp.0` | Linear(256 $\to$ 256) | 65,536 | 256 | 65,792 |

| `fusion_mlp.1` | LayerNorm(256) | 256 | 256 | 512 |

| `fusion_mlp.2,3` | ReLU + Dropout(0.2) | 0 | 0 | 0 |

| `fusion_mlp.4` | Linear(256 $\to$ 128) | 32,768 | 128 | 32,896 |

| `fusion_mlp.5` | LayerNorm(128) | 128 | 128 | 256 |

| `head.weight` | Shared Projection Linear(1, 128) | 128 | 0 | 128 |

| `head.theta_0` | Base Cutpoint Parameter (1) | 1 | 0 | 1 |

| `head.raw_deltas` | Cutpoint Decrements (3) | 3 | 0 | 3 |

| **TOTAL TRAINABLE** | **Full Monotonic CORAL Head** | **623,618** | **2,562** | **626,180** |


> **Resolution of Parameter Claim:** The exact trainable parameter count is **626,180** (~626k parameters), fully verified from state dict tensors.


---

## 2. Mathematical Proof of Structural Monotonicity


To prevent threshold inversion artifacts common in unconstrained ordinal classification, the `MonotonicCORALHead` enforces monotonicity by construction:

1. **Logit Projection:**

   $$\text{logits}_k = x W^T + b_k, \quad k \in \{0, 1, \dots, K-2\}$$

2. **Strictly Decreasing Threshold Biases:**

   $$b_0 = \theta_0$$

   $$b_k = b_{k-1} - \text{softplus}(\delta_k), \quad k = 1, \dots, K-2$$

   Because $\text{softplus}(\delta) = \ln(1 + e^\delta) > 0$ strictly for all $\delta \in \mathbb{R}$, each decrement is strictly positive:

   $$b_0 > b_1 > b_2 > \dots > b_{K-2}$$

3. **Monotonic Cumulative Logits:**

   $$\text{logits}_0 > \text{logits}_1 > \dots > \text{logits}_{K-2} \implies P(Y > 0) > P(Y > 1) > \dots > P(Y > K-2)$$

4. **Valid Discrete Probability Construction:**

   $$P(Y = 0) = 1 - P(Y > 0)$$

   $$P(Y = k) = P(Y > k-1) - P(Y > k) \ge 0$$

   $$P(Y = K-1) = P(Y > K-2)$$

   $$\sum_{k=0}^{K-1} P(Y = k) = 1 - P(Y > 0) + \sum_{k=1}^{K-2} [P(Y > k-1) - P(Y > k)] + P(Y > K-2) \equiv 1.0$$

5. **Canonical Ordinal Decision Rule:**

   $$\hat{y} = \sum_{k=0}^{K-2} \mathbb{I}(P(Y > k) > 0.5) \in \{0, 1, \dots, K-1\}$$

   Because $P(Y > k)$ is strictly monotonic, threshold crossings form a single contiguous interval, guaranteeing that rank decoding is identical to cumulative thresholding without heuristic post-processing.


---

## 3. Training Protocol & Hyperparameter Specifications


| Hyperparameter / Setting | Verified Value | Justification / Notes |

|:---|:---|:---|

| **Optimizer** | AdamW | Decoupled weight decay regularization |

| **Learning Rate** | $1.0 \times 10^{-3}$ | Fixed learning rate for 15-epoch convergence |

| **Weight Decay** | $1.0 \times 10^{-4}$ | Standard $L_2$ penalty on weights |

| **Batch Size** | 2,048 | Large batch size feasible via precomputed 768-D embeddings |

| **Training Epochs** | 15 | Head converges within 10-12 epochs |

| **Loss Function** | Weighted Binary Cross-Entropy | Sum of binary ordinal tasks: $\mathcal{L} = \sum_{k} w_k \text{BCE}(h_k, y_k)$ |

| **Class Imbalance Weighting** | Effective Number (Cui et al., 2019) | $\beta = 0.999$, weights inversely proportional to $(1 - \beta^{n_k}) / (1 - \beta)$ |

| **Checkpoint Selection** | Best Validation QWK | Model with peak validation Quadratic Weighted Kappa saved as `best` |

| **Random Seeds** | 30 prespecified seeds | `[42, 123, 777, 999, 2025, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]` |


---

## 4. Empirical Hardware Latency & Throughput Benchmark


- **Benchmark Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.1)

- **Framework:** PyTorch 2.5.1+cu121 with CUDA synchronization (`torch.cuda.synchronize()`) and 50 warm-up iterations.


| Scope | Precision | Batch Size | Latency / Batch (ms) | Latency / Sample (ms) | Throughput (FPS) | Primary Computational Bottleneck |

|:---|:---:|---:|---:|---:|---:|:---|

| **Head-Only** | FP32 | 1 | 1.27 ms | 1.269 ms | **787.7** | GPU kernel dispatch overhead |

| **Head-Only** | FP32 | 16 | 1.26 ms | 0.079 ms | **12,655.8** | Matrix multiplication |

| **Head-Only** | FP32 | 32 | 2.57 ms | 0.080 ms | **12,441.8** | Matrix multiplication |

| **Head-Only** | FP32 | 2,048 | 6.77 ms | 0.003 ms | **302,417.9** | Memory bandwidth saturated |

| **Head-Only** | FP16 | 1 | 1.38 ms | 1.381 ms | **724.0** | Kernel dispatch |

| **Head-Only** | FP16 | 16 | 1.11 ms | 0.069 ms | **14,454.0** | Tensor core execution |

| **Head-Only** | FP16 | 32 | 1.30 ms | 0.041 ms | **24,618.6** | Tensor core execution |

| **Full Pipeline (E2E)** | FP32 | 1 | 37.10 ms | 37.100 ms | **27.0** | ConvNeXt-Tiny depthwise convs |

| **Full Pipeline (E2E)** | FP32 | 16 | 404.36 ms | 25.272 ms | **39.6** | ConvNeXt-Tiny FLOPs |

| **Full Pipeline (E2E)** | FP32 | 32 | 797.29 ms | 24.915 ms | **40.1** | ConvNeXt-Tiny FLOPs |

| **Full Pipeline (E2E)** | **FP16** | **1** | **20.43 ms** | **20.428 ms** | **49.0** | ConvNeXt-Tiny Tensor Core FP16 |

| **Full Pipeline (E2E)** | **FP16** | **16** | **180.64 ms** | **11.290 ms** | **88.6** | ConvNeXt-Tiny Tensor Core FP16 |

| **Full Pipeline (E2E)** | **FP16** | **32** | **354.08 ms** | **11.065 ms** | **90.4** | ConvNeXt-Tiny Tensor Core FP16 |



### Deployment Claims & Boundary Conditions:

1. **Unit Clarification:** In early working drafts, milliseconds per batch and milliseconds per sample were occasionally conflated. The authoritative table explicitly separates latency per batch from latency per sample.

2. **Head-Only vs Full Pipeline:** Head-only inference exceeds 10,000 FPS at BS $\ge 16$ and reaches 300,000 FPS at batch size 2,048. However, complete image-to-decision deployment requires the ConvNeXt-Tiny backbone, delivering **49.0 FPS at BS=1** and **90.4 FPS at BS=32** in FP16.

3. **Commercial Readiness Bounds:** While 49.0 to 90.4 FPS comfortably exceeds mechanical conveyor sorting speeds (typically 10 to 30 fruit/s per lane), true commercial deployment requires real-time hardware triggering, optical illumination normalization, and handling physical fruit rotation and dirt occlusion, which are outside the scope of this benchmark.

