# Phase 5.6 — Real-Image Open-Set OOD Forensic Audit

**Audit Timestamp:** 2026-10-14 13:46:36  

**Evaluation Scope:** Forensic revalidation of open-set recognition on 5,000 real images of unencountered fruit species (Cherry, Kiwi, Peach, Pineapple, Plum) vs 12,833 in-distribution test images.

---

## 1. Resolution of Historical Synthetic Feature Defect


> **Historical Defect:** Prior to Phase 5.0, an unrealistic 100.0% AUROC claim was reported in draft notes. Forensic inspection revealed this was an artifact of synthetic Gaussian noise rather than real images.

> **Definitive Fix:** All reported metrics evaluate strictly on **5,000 real images** extracted through the frozen ConvNeXt-Tiny backbone (`outputs/casm_v2/phase50_ood_features/ood_real_features.pt`, 15.8 MB). Synthetic features have been permanently expunged.


---

## 2. Authoritative Pooled OOD Benchmark Performance


| Detector | Feature Representation | AUROC (%) | 95% Bootstrap CI | AUPR (OOD=1) (%) | FPR @ 95% TPR (%) | Parameter Fitting Protocol |

|:---|:---|---:|:---:|---:|---:|:---|

| **Mahalanobis Distance** | ConvNeXt-Tiny 768-D Embeddings | **84.22%** | [83.7%, 84.8%] | **53.28%** | **38.39%** | Training partition only (class means & tied precision) |

| **Free Energy Score** | Ordinal Head Logits (T=1.0) | **81.99%** | [81.3%, 82.6%] | **49.96%** | **41.38%** | Validation thresholding at 95% in-dist TPR |

| **Maximum Softmax Probability (MSP)** | Softmax Class Probabilities | **74.26%** | [73.6%, 74.9%] | **43.16%** | **47.85%** | Validation thresholding at 95% in-dist TPR |



- **Positive Class Definition:** $y=1$ denotes OOD samples (unseen fruit species), and $y=0$ denotes in-distribution test samples. AUPR is computed strictly with OOD as the positive class.

- **Score Direction Convention:** For all detectors, higher scores strictly indicate greater OOD likelihood ($+\text{dist}$ for Mahalanobis, $-T \log \sum e^{z/T}$ for Energy, and $-\max_k P(Y=k)$ for MSP).


---

## 3. Per-Species Breakdown Across 5 Unencountered Fruits


| Unseen Species | Sample N | Mahalanobis AUROC | Energy AUROC | MSP AUROC |

|:---|---:|---:|---:|---:|

| **Cherry** | 1,000 | **76.99%** | **81.02%** | 69.06% |

| **Kiwi** | 1,000 | **84.77%** | **82.84%** | 79.12% |

| **Peach** | 1,000 | **78.28%** | **74.21%** | 71.98% |

| **Pineapple** | 1,000 | **93.00%** | **84.23%** | 74.94% |

| **Plum** | 1,000 | **88.08%** | **87.64%** | 76.21% |


---

## 4. Domain-Dependent Detector Inversion & Empirical Explanation


When stratified across acquisition domains, a dramatic performance divergence emerges:


| In-Distribution Domain | Sample N | Proportion (%) | Mahalanobis AUROC | Energy AUROC | MSP AUROC |

|:---|---:|---:|---:|---:|---:|

| **fruits360 (Clean Studio Backgrounds)** | 10,974 | 85.51% | **93.26%** | **79.99%** | **75.21%** |

| **real_grade (Industrial Conveyors)** | 67 | 0.52% | **18.81%** | **98.89%** | **96.57%** |

| **real_quality (Market Bins)** | 1,480 | 11.53% | **18.08%** | **92.49%** | **61.74%** |

| **real_severity (Controlled Lab Images)** | 312 | 2.43% | **94.10%** | **98.67%** | **95.30%** |



### Scientific Characterization of Detector Reversals:

1. **Mahalanobis Collapse on Real-Market Bins (18.08%) and Conveyors (18.81%):**

   - Mahalanobis distance measures feature deviation from the training centroid and covariance. Because the training set is ~85.5% studio images with clean white backgrounds, feature covariance is conditioned on studio characteristics.

   - Real-market fruits (`real_quality`) and industrial conveyor images (`real_grade`) contain complex backgrounds, produce bins, and conveyor rollers. The feature extractor projects these into regions distant from the studio centroid.

   - As an empirical result, in-distribution market images receive *larger* Mahalanobis distances than OOD studio fruits, completely inverting detector ordering.

2. **Energy Score Robustness (92.49% on Market, 98.89% on Conveyor):**

   - Free Energy aggregates unnormalized logits and is less sensitive to pure feature-norm inflation. Even when background shift alters raw embedding norms, the ordinal classifier produces lower overall logit energy for unencountered classes.

3. **Non-Causal Attribution Statement:**

   - Although background complexity strongly correlates with the observed detector failure, the manuscript strictly describes this as an *observed distribution-dependent performance inversion* rather than claiming background causality, as controlled background-swap ablation experiments were not performed.

