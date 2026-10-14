# Phase 5.6 — Calibration & Selective Prediction Forensic Audit

**Audit Timestamp:** 2026-10-14 14:17:34  

**Audit Objective:** Disentangle single-seed exemplar metrics, multi-seed averages, and ensemble calibration; establish canonical ordinal cutpoint vs categorical argmax baseline MAE; document domain stratification.

---

## 1. Disentangling Calibration Metrics


Historical documents cited ECE values of **1.47%**, **2.11%**, and **2.51%** without consistently clarifying their distinct mathematical scopes:


| Scope | Evaluation Protocol | Test N | Observed Accuracy | Ordinal MAE | ECE (10-bin) | Brier Score | Negative Log-Likelihood |

|:---|:---|---:|:---:|:---:|:---:|:---:|:---:|

| **Single Exemplar Seed 83** | Individual Checkpoint | 12833 | **93.71%** | **0.0889** | **1.47%** | 0.0857 | 0.1736 |

| **30-Seed Independent Aggregate (Mean ± SD)** | Per-Seed Summary Statistics | 12833 | **92.11 ± 1.49%** | **0.1145 ± 0.0179** | **2.11 ± 0.53%** | 0.0940 ± 0.0065 | 0.1940 ± 0.0141 |

| **30-Seed Probability Ensemble** | Averaged Probability Distribution | 12833 | **91.15%** | **0.1382** | **2.51%** | 0.0887 | 0.1834 |



> **Resolution:**

> - **1.47% ECE / 0.0857 Brier:** Derived from **Seed 83 alone** (`preds_A_seed83.pt`).

> - **2.11% ± 0.53% ECE / 0.0940 Brier:** The true **30-seed aggregate mean ± standard deviation** across all independently trained models.

> - **2.51% ECE / 0.0887 Brier:** Derived from the **30-seed probability ensemble** (averaging predicted class distributions prior to calibration binning).


---

## 2. Ordinal Cutpoint vs Argmax Baseline Reconciliation


The widely reported **84.7% error reduction** at 90% coverage has been rigorously verified:

$$\text{Relative Reduction} = \frac{\text{Baseline MAE} - \text{Selective MAE}}{\text{Baseline MAE}} \times 100\%$$


| Metric / Feature | Ordinal Cumulative Cutpoint Rule $\sum \mathbb{I}(h_k > 0.5)$ | Categorical Argmax Rule $\operatorname{argmax}(P)$ |

|:---|:---:|:---:|

| **Baseline Accuracy (100% Coverage)** | **93.71%** | 91.98% |

| **Baseline MAE (100% Coverage)** | **0.0889** (Matches Checkpoint) | 0.1234 |

| **Selective Accuracy (90% Coverage)** | **99.28%** | 99.28% |

| **Selective MAE (90% Coverage)** | **0.0136** | 0.0136 |

| **Exact Relative Error Reduction (%)** | **84.71%** (Exact 84.7%) | 88.98% |

| **Concordance with Stored Model Metric** | **EXACT MATCH (`pt['mae'] == 0.0889`)** | Mismatch (post-hoc categorical argmax) |



### Arithmetic Demonstration:

$$\text{Ordinal Reduction} = \frac{0.0889 - 0.0136}{0.0889} = \frac{0.0753}{0.0889} = 0.847019 \implies \mathbf{84.70\%}$$

$$\text{Argmax Reduction} = \frac{0.1234 - 0.0136}{0.1234} = \frac{0.1098}{0.1234} = 0.889789 \implies \mathbf{88.98\%}$$


> **Resolution:** MonotonicCORAL is an ordinal regression model whose canonical decision rule is threshold-based rank counting $\sum_{k} \mathbb{I}(h_k > 0.5)$. The baseline MAE is strictly **0.0889**, and the exact reduction to **0.0136** is **84.70%**. Previous drafts inadvertently paired 0.1234 with 84.7%.


---

## 3. Selective Abstention Profile Across Coverage Levels (Seed 83)


| Target Coverage | Retained N | Rejected N | Rejection Rate (%) | Min Confidence Threshold | Selective Accuracy (%) | Selective MAE | Relative Error Reduction (%) |

|:---:|---:|---:|---:|:---:|:---:|:---:|:---:|

| **100%** | 12,833 | 0 | 0.00% | $\ge$ 0.2429 | **93.71%** | **0.0889** | **0.00%** |

| **95%** | 12,191 | 642 | 5.00% | $\ge$ 0.2950 | **97.28%** | **0.0451** | **49.26%** |

| **90%** | 11,550 | 1,283 | 10.00% | $\ge$ 0.6431 | **99.28%** | **0.0136** | **84.71%** |

| **85%** | 10,908 | 1,925 | 15.00% | $\ge$ 0.9333 | **99.92%** | **0.0023** | **97.42%** |

| **80%** | 10,266 | 2,567 | 20.00% | $\ge$ 0.9936 | **100.00%** | **0.0000** | **100.00%** |

| **70%** | 8,983 | 3,850 | 30.00% | $\ge$ 0.9988 | **100.00%** | **0.0000** | **100.00%** |

| **60%** | 7,700 | 5,133 | 40.00% | $\ge$ 0.9991 | **100.00%** | **0.0000** | **100.00%** |

| **50%** | 6,416 | 6,417 | 50.00% | $\ge$ 0.9992 | **100.00%** | **0.0000** | **100.00%** |



Across all 30 seeds, 90% coverage yields:

- **Mean Selective Accuracy:** 98.79% ± 0.66%

- **Mean Selective MAE:** 0.0221 ± 0.0076

- **Mean Error Reduction:** **81.02%**


---

## 4. Domain-Stratified Calibration Under Severe Imbalance


The pooled test set ($N=12,833$) is heavily dominated by clean studio turntable images (`fruits360`), representing **85.5%** of all test samples:


| Acquisition Domain | Sample Count | Test Proportion (%) | Accuracy (%) | Ordinal MAE | ECE (%) | Brier Score | NLL |

|:---|---:|---:|---:|---:|---:|---:|---:|

| **fruits360** | 10,974 | 85.51% | 99.77% | 0.0024 | 0.83% | 0.0028 | 0.0099 |

| **real_grade** | 67 | 0.52% | 22.39% | 1.194 | 37.16% | 1.0359 | 2.096 |

| **real_quality** | 1,480 | 11.53% | 54.8% | 0.6581 | 10.29% | 0.6243 | 1.2471 |

| **real_severity** | 312 | 2.43% | 80.45% | 0.1955 | 14.9% | 0.2433 | 0.4268 |



> **Critical Scientific Caveat:**  

> While the pooled test ECE is low (1.47% - 2.51%), this is overwhelmingly driven by the 10,974 clean studio images (`fruits360`, ECE 0.83%). When evaluated on real market bins (`real_quality`, ECE 10.29%) or industrial conveyor frames (`real_grade`, ECE 37.16%), the model exhibits substantial miscalibration. Claims of universal 99% accuracy or low calibration error must be strictly qualified by domain.

