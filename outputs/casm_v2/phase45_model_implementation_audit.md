# Phase 4.5 — Model Implementation and Architecture Audit Report

**Auditors:** Senior PyTorch Architecture Auditor, Ordinal Regression Specialist, Machine Learning Research Engineer  
**Date:** October 2026  
**Repository:** `C:\fruit_dataset`  
**Deliverables:**
- `outputs/casm_v2/phase45_model_implementation_audit.md`
- `outputs/casm_v2/casm_v2_corrected_modules.py`
- `outputs/casm_v2/test_corrected_modules.py` (Unit tests passed: 4/4)

---

## 1. Executive Summary

This audit evaluates the deep learning architecture implementations in the repository, focusing on:
1. Mathematical validity and probability coherence of the Consistent Rank Logits (CORAL) ordinal regression head.
2. Species encoding integrity and unknown-species open-set fallback mechanisms.
3. Rigorous parameter count reconciliation across all ablation variants.

### Key Audit Findings:
1. **CORAL Threshold Unconstrained Defect (Task 3.1):** The existing implementation (`CORALHead` in `context_audit_train.py` and `severity_model.py`) parameterizes threshold cutpoints as an unconstrained bias vector $\mathbf{b} \in \mathbb{R}^{K-1}$. While empirical training happened to preserve $b_0 \ge b_1 \ge b_2 \ge b_3$ across the 45 Phase 4.3 checkpoints, the model provides **no structural guarantee of threshold monotonicity**. Under perturbed or inverted biases, cumulative probabilities invert ($P(Y > k) > P(Y > k-1)$), producing **negative class probabilities** ($P(Y=k) < 0$). We implemented and unit-tested `MonotonicCORALHead` in `casm_v2_corrected_modules.py`, which guarantees strictly decreasing thresholds via positive softplus decrements.
2. **Species ID Out-of-Bounds Crash & Lack of Fallback (Task 3.2):** `AuditModel` instantiates `nn.Embedding(5, 16)`, supporting only known IDs $0..4$. Inputting out-of-range IDs ($\ge 5$) or negative IDs raises an unhandled `IndexError: index out of range in self`. There is **no reserved unknown-species index**. We implemented `RobustSpeciesEmbedding` with 6 categories (index 5 explicitly reserved as `UNKNOWN_SPECIES`) and automatic sanitization for open-set inference.
3. **Parameter Count Reconciliation (Task 3.3):** The historical claim in `phase44_reproducibility_report.md` that Config A had 772 parameters and Config B had 238,629 parameters was completely inaccurate. The actual instantiated `AuditModel` possesses **626,180 parameters** for Config A and **630,356 parameters** for Config B, due to a 2-layer visual projection MLP (526,592 parameters) and a 2-layer fusion MLP (99,456–103,552 parameters).

---

## 2. Audit of CORAL Ordinal Regression (Task 3.1)

### Mathematical Formulation & Axiomatic Requirements:
In $K$-class ordinal regression ($y \in \{0, 1, \dots, K-1\}$), CORAL decomposes multi-class prediction into $K-1$ binary classification subtasks:
$$y^{(k)} = \mathbb{I}(y > k), \quad k \in \{0, 1, \dots, K-2\}$$
The predicted cumulative probability is:
$$\hat{P}(Y > k \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b_k)$$
By the Kolmogorov probability axioms, since event $\{Y > k+1\} \subseteq \{Y > k\}$:
$$P(Y > k) \ge P(Y > k+1) \quad \forall \mathbf{x}$$
Because the logistic sigmoid $\sigma(z)$ is strictly monotonically increasing, this condition holds if and only if:
$$\mathbf{w}^T \mathbf{x} + b_k \ge \mathbf{w}^T \mathbf{x} + b_{k+1} \iff b_k \ge b_{k+1} \quad \forall k$$

### Structural Flaw in Existing Implementation:
In `context_audit_train.py` (lines 65–76):
```python
class CORALHead(nn.Module):
    def __init__(self, in_features=128, num_classes=5):
        super().__init__()
        self.num_tasks = num_classes - 1
        self.weight = nn.Parameter(torch.Tensor(1, in_features))
        self.bias = nn.Parameter(torch.zeros(self.num_tasks)) # UNCONSTRAINED!
```
The parameter `self.bias` is a completely unconstrained tensor in $\mathbb{R}^4$. Nothing prevents AdamW gradient updates from producing $b_1 > b_0$.

### Impact on Discrete Class Probabilities:
Discrete class probabilities are recovered via telescoping subtraction:
$$P(Y = 0) = 1 - P(Y > 0)$$
$$P(Y = k) = P(Y > k-1) - P(Y > k) \quad \text{for } 1 \le k \le K-2$$
$$P(Y = K-1) = P(Y > K-2)$$
If $b_1 > b_0$, then for any input $\mathbf{x}$:
$$P(Y > 1) > P(Y > 0) \implies P(Y = 1) = P(Y > 0) - P(Y > 1) < 0$$
We demonstrated this failure mode in `test_corrected_modules.py`: setting $\mathbf{b} = [-1.0, 1.5, 0.5, -0.5]$ produced **$P(Y=1) = -0.5486$**, violating probability theory.

### Empirical Check of Existing Checkpoints:
We inspected all 45 saved checkpoints in `outputs/phase4/audit/`:
- Checkpoints with non-monotonic biases: **0 / 45**.
- In this specific dataset and training run, the binary cross-entropy loss gradients happened to push $b_0 > b_1 > b_2 > b_3$ organically.
- However, relying on empirical convergence without architectural constraints is unsafe for safety-critical agricultural inspection and IEEE Transactions publication.

### Corrected Implementation (`MonotonicCORALHead`):
We implemented cutpoint reparameterization in `casm_v2_corrected_modules.py`:
$$b_0 = \theta_0, \quad b_k = b_{k-1} - \text{softplus}(\delta_k), \quad k = 1, \dots, K-2$$
Because $\text{softplus}(\delta) = \ln(1 + e^\delta) > 0$ for all $\delta \in \mathbb{R}$, $b_k < b_{k-1}$ is guaranteed unconditionally.
- Unit test verification:
  * Biases strictly decreasing: **PASSED**
  * Cumulative probabilities non-increasing: **PASSED**
  * Discrete class probabilities $P(Y=k) \ge 0$: **PASSED**
  * Probability sum $\sum_k P(Y=k) \equiv 1.0000 \pm 1e-5$: **PASSED**
  * Extreme inputs ($\pm 1000.0$) numerical stability: **PASSED**
  * FP16 mixed precision inference: **PASSED**

---

## 3. Audit of Species Encoding and Unknown-Species Fallback (Task 3.2)

### Existing Implementation Flaw:
In `context_audit_train.py` (lines 88):
```python
self.species_embed = nn.Embedding(num_species, 16) # num_species = 5
```
1. Known species IDs: `0: Apple, 1: Banana, 2: Mango, 3: Orange, 4: Tomato`.
2. When evaluating open-set fruit images (e.g. Lemon, Peach, Dragonfruit) or corrupted metadata:
   - Input ID `5` raises `IndexError: index out of range in self`.
   - Input ID `-1` raises `IndexError: index out of range in self`.
3. The model **cannot execute inference on unknown species**, rendering open-set deployment impossible without crashing.

### Corrected Implementation (`RobustSpeciesEmbedding`):
Implemented in `casm_v2_corrected_modules.py`:
- `num_categories = 6`: Indices $0..4$ correspond to known fruits; index $5$ is explicitly reserved for `UNKNOWN_SPECIES`.
- Method `sanitize_ids(ids)` automatically routes any ID $< 0$ or $\ge 5$ to index $5$.
- Unit test verification:
  * Known IDs $0..4$ preserve exact identity: **PASSED**
  * Negative IDs ($-1, -10, -999$) map to index 5: **PASSED**
  * Out-of-bounds IDs ($5, 6, 1000$) map to index 5: **PASSED**
  * Single-image and batched forward passes identical: **PASSED**

---

## 4. Parameter Count Reconciliation (Task 3.3)

We computed exact parameter counts for all model variants using PyTorch tensor inspection:

### Table 1: Exact Instantiated Parameter Counts (`AuditModel` used in Phase 4.3 & 4.4)
| Configuration | Model Name | Visual Projection MLP | Species Embedding | Fusion MLP | CORAL Head | Total Trainable Parameters |
|:---:|---|---:|---:|---:|---:|---:|
| **Config A** | Image Only | 526,592 | 0 | 99,456 | 132 | **626,180** |
| **Config B** | Image + Species | 526,592 | 80 | 103,552 | 132 | **630,356** |
| **Config C** | Image + OOD | 526,592 | 0 | 99,968 | 132 | **626,692** |
| **Config D** | Image + Species + OOD | 526,592 | 80 | 104,064 | 132 | **630,868** |

### Breakdown of Layer Dimensions:
- `vis_proj`:
  * `Linear(768, 512)`: $393,728$ params
  * `LayerNorm(512)`: $1,024$ params
  * `Linear(512, 256)`: $131,328$ params
  * `LayerNorm(256)`: $512$ params
  * Subtotal: **526,592 parameters**
- `species_embed`:
  * `Embedding(5, 16)`: **80 parameters**
- `fc` (Fusion MLP):
  * Config A: `Linear(256, 256)` ($65,792$) + `LN(256)` ($512$) + `Linear(256, 128)` ($32,896$) + `LN(128)` ($256$) = **99,456 parameters**
  * Config B: `Linear(272, 256)` ($69,888$) + `LN(256)` ($512$) + `Linear(256, 128)` ($32,896$) + `LN(128)` ($256$) = **103,552 parameters**
  * Config C: `Linear(258, 256)` ($66,304$) + `LN(256)` ($512$) + `Linear(256, 128)` ($32,896$) + `LN(128)` ($256$) = **99,968 parameters**
  * Config D: `Linear(274, 256)` ($70,400$) + `LN(256)` ($512$) + `Linear(256, 128)` ($32,896$) + `LN(128)` ($256$) = **104,064 parameters**
- `head` (`CORALHead`):
  * `weight`: $1 \times 128 = 128$ params
  * `bias`: $4$ params
  * Subtotal: **132 parameters**

---

### Comparison with Backbone & Historical Claims:
| Model / Report Citation | Claimed Parameters | Actual Audited Parameters | Discrepancy Reason |
|---|---:|---:|---|
| **Phase 1 ConvNeXt-Tiny Backbone** | ~28 M | **27,823,973** | Exact match with timm convnext_tiny specifications. |
| **Phase 4.4 Report Config A** | 772 | **626,180** | Report author only counted Linear(768, 1)+4, ignoring the 526k vis_proj and 99k fc MLPs. |
| **Phase 4.4 Report Config B** | 238,629 | **630,356** | Report author described a hypothetical 800->256->128 MLP that was not in the codebase. |
| **Phase 4.4 Report Config D** | 239,141 | **630,868** | Same hypothetical MLP specification in report text. |
| **Phase 4.0 CASM v1 Mode A** | ~560k | **559,956** | Matched ContextAwareSeverityNetwork code. |
| **Phase 4.0 CASM v1 Mode E** | ~974k | **974,356** | Matched GatedContextFusion + FiLM architecture. |

---

## 5. Stage 3 Sign-Off Verdict

- **Inspected:** `context_audit_train.py`, `severity_model.py`, 45 checkpoint tensors, parameter computation scripts.
- **Verified:** Parameter counts across all 4 configs; empirical monotonicity of the 45 Phase 4.3 checkpoints; confirmed failure mode of unconstrained bias mathematically and empirically; confirmed missing unknown-species fallback.
- **Artifacts Generated:**
  * `casm_v2_corrected_modules.py`: Structurally monotonic CORAL head and robust species embedding.
  * `test_corrected_modules.py`: 4 unit tests passing 100%.
- **Safety to Proceed:** It is **SAFE** to proceed to Phase 4 (Reproduce Statistical Results).
