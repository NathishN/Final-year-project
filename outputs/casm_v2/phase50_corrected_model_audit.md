# Phase 5.0 -- Stage 2: Corrected Model Integration Audit

**Date:** October 2026  
**Stage:** 2 -- Integrate and Test Corrected Model Components  
**Unit Tests:** 18/18 passed  
**Smoke Test:** PASSED

---

## 1. Implementation Change Summary

### MonotonicCORALHead (replacing CORALHead)

| Property | Old CORALHead | MonotonicCORALHead |
|---|---|---|
| Threshold parameterization | Unconstrained bias vector | Reparameterized: theta_0 - cumsum(softplus(delta)) |
| Monotonicity guarantee | None (empirical only) | Structural: softplus > 0 guarantees b_k > b_{k+1} |
| Negative class probs possible | YES | NO (clamped + normalized) |
| Probability sum | Not guaranteed | Guaranteed (normalized) |
| Extreme input stability | Unverified | Verified (FP32 + FP16) |

### RobustSpeciesEmbedding (replacing nn.Embedding(5, 16))

| Property | Old nn.Embedding | RobustSpeciesEmbedding |
|---|---|---|
| Known species | 5 (0-4) | 5 (0-4) |
| Unknown species index | None (crashes) | Index 5 explicitly reserved |
| Negative ID handling | IndexError crash | Routes to index 5 |
| Out-of-range handling | IndexError crash | Routes to index 5 |
| Validation at inference | None | sanitize_ids() called automatically |

---

## 2. CORAL Mathematical Consistency Audit

### Training Loss Definition
The CORAL training loss uses binary cross-entropy on K-1 ordinal tasks:
```
L = sum_{k=0}^{K-2} BCE(sigma(logit_k), I(y > k))
```
For y=2 with K=5: binary targets = [1, 1, 0, 0]

### Inference Decoding Definition
Predicted rank = count(sigma(logit_k) > 0.5) = count(P(Y>k) > 0.5)

### Consistency Verification
Both training loss and inference decoding operate on the same logit tensor.
The MonotonicCORALHead.forward() returns logits [B, K-1].
Training: F.binary_cross_entropy_with_logits(logits, binary_targets)
Inference: (torch.sigmoid(logits) > 0.5).sum(dim=1)
These are numerically consistent -- same logits, same sigmoid operation.

### Cutpoint Construction Audit
```
b_0 = theta_0
b_k = theta_0 - sum_{j=1}^{k} softplus(delta_j)  for k=1,...,K-2
```
softplus(x) = log(1 + exp(x)) > 0 for all x in R
Therefore: b_0 > b_1 > ... > b_{K-2} UNCONDITIONALLY
Therefore: P(Y>0) > P(Y>1) > ... > P(Y>K-2) UNCONDITIONALLY
Therefore: P(Y=k) = P(Y>k-1) - P(Y>k) >= 0 UNCONDITIONALLY

---

## 3. Unit Test Results

**Total:** 18 tests  
**Passed:** 18  
**Failed:** 0  
**Errors:** 0  
**Status:** PASSED

### Test Details

**MonotonicCORALHead Tests:**
- test_output_shape: logits=[B,4] verified
- test_bias_strictly_decreasing: verified across 10 random seeds
- test_cumulative_prob_monotonic: verified on 64 samples
- test_class_probs_nonnegative: verified on 64 samples
- test_class_probs_sum_to_one: max_err < 1e-5
- test_class_probs_shape: [B,5] verified
- test_rank_valid_range: ranks in {0,1,2,3,4}
- test_extreme_logits_no_nan: extreme +/-1000 inputs stable
- test_mixed_precision_fp16: FP16 forward pass valid
- test_loss_inference_consistency: training and inference use same logits
- test_backward_gradient_finiteness: all parameter gradients finite

**RobustSpeciesEmbedding Tests:**
- test_known_species_pass_through: IDs 0-4 pass through unchanged
- test_unknown_index_five_reserved: index 5 stays as 5
- test_negative_ids_fallback: all negative -> 5
- test_out_of_range_ids_fallback: >4 -> 5
- test_mixed_batch_routing: mixed batch produces correct unknown embeddings
- test_output_shape: [B, embedding_dim] correct
- test_training_inference_identical_encoding: train == eval for same IDs

---

## 4. Smoke Test Results

**Status:** PASSED  
**Model Parameters:** 630,372  
**Training Loss:** 0.7828  
**Gradient Issues:** []  
**Weight Ratio (EN weights, beta=0.999):** 5.32x

### Effective Number of Samples Class Weights

| Grade | Name | Train N | EN Weight |
|:---:|---|---:|---:|
| 0 | Healthy | 36,414 | 0.001000 |
| 1 | Mild | 846 | 0.001751 |
| 2 | Moderate | 208 | 0.005323 |
| 3 | Severe | 1,606 | 0.001251 |
| 4 | Rotten | 716 | 0.001955 |

Weight ratio (Grade 2 / Grade 0): 5.32x
Weight ratio (max / min): 5.32x  
(Compare: original inverse weighting had 111x ratio; EN weighting reduces this significantly)

---

## 5. Stage 2 Sign-Off

- Unit Tests: 18/18 passed
- Smoke Test: PASSED
- CORAL threshold ordering mathematically verified
- Probability non-negativity guaranteed by clamp + normalize
- Species unknown-species fallback verified
- Training loss and inference decoding numerically consistent
- Effective Number of Samples weights computed from split_v2 train counts

**SAFE TO PROCEED TO STAGE 3 (Loss Weighting & Pilot)**