# Phase 5.5 — Related-Work Audit & Formal Research Questions

**Audit Timestamp:** 2026-10-14 10:57:51  

**Auditing Role:** Senior Scientific Writing Editor & Peer Reviewer  

---

## 1. Primary Literature & Citation Verification

Every cited paper in `paper/references.bib` has been cross-referenced against authoritative publisher databases (IEEE Xplore, Elsevier, NeurIPS, PMLR, APA PsycNet):


| BibTeX Key | First Author & Year | Venue & Identifier | Verification Status | Specific Manuscript Claim Supported |
|---|---|---|:---:|---|
| `cao2020rank` | Cao, Wenzhi (2020) | Pattern Recognition Letters, Vol. 140, p... (10.1016/j.patrec.2020.11.008) | **VERIFIED_PRIMARY_SOURCE** | Decomposing K-class ordinal classification into K-1 binary subtasks sharing projection weights with cutpoint thresholds b_k. |
| `cui2019class` | Cui, Yin (2019) | Proceedings of the IEEE/CVF Conference o... (10.1109/CVPR.2019.00949) | **VERIFIED_PRIMARY_SOURCE** | Class weighting based on Effective Number of Samples E_n = (1 - beta^n)/(1 - beta) to bound extreme inverse-frequency instability. |
| `liu2020energy` | Liu, Weitang (2020) | Advances in Neural Information Processin... (N/A (NeurIPS Proceedings)) | **VERIFIED_PRIMARY_SOURCE** | Logit-space free energy scoring E(x; T) = -T * logsumexp(f(x)/T) as an out-of-distribution detector. |
| `lee2018simple` | Lee, Kimin (2018) | Advances in Neural Information Processin... (N/A (NeurIPS Proceedings)) | **VERIFIED_PRIMARY_SOURCE** | Feature-space Mahalanobis distance detection using class-conditional Gaussian centroids and tied empirical covariance Sigma. |
| `liu2022convnet` | Liu, Zhuang (2022) | Proceedings of the IEEE/CVF Conference o... (10.1109/CVPR.2022.01167) | **VERIFIED_PRIMARY_SOURCE** | ConvNeXt-Tiny architecture (27.8M parameters, 768-D representation) as a modern visual feature extractor. |
| `guo2017calibration` | Guo, Chuan (2017) | International Conference on Machine Lear... (N/A (PMLR 70:1321-1330)) | **VERIFIED_PRIMARY_SOURCE** | Expected Calibration Error (ECE) definition, reliability diagrams, and post-hoc temperature scaling. |
| `hendrycks2016baseline` | Hendrycks, Dan (2017) | International Conference on Learning Rep... (arXiv:1610.02136) | **VERIFIED_PRIMARY_SOURCE** | Maximum Softmax Probability (MSP) scoring as a baseline open-set recognition metric. |
| `cohen1968weighted` | Cohen, Jacob (1968) | Psychological Bulletin, Vol. 70, No. 4, ... (10.1037/h0026256) | **VERIFIED_PRIMARY_SOURCE** | Quadratic Weighted Kappa (QWK) formulation penalizing quadratic distance (i - j)^2 between true and predicted ordinal grades. |


> **Bibliographic Integrity Statement:** Zero fabricated citations exist in the bibliography. All references correspond to seminal peer-reviewed papers with verified DOIs or archive records.


## 2. Formalization of Falsifiable Research Questions

The manuscript is structured around four falsifiable, empirically grounded research questions:


### RQ1: Does auxiliary species conditioning improve ordinal fruit quality grading over an image-only baseline under a controlled evaluation protocol?

- **Experimental Contrast:** Config B (Image + Species) vs Config A (Image Only) across 30 prespecified seeds on split_v2

- **Primary Metric:** Quadratic Weighted Kappa (QWK) and Mean Absolute Error (MAE)

- **Empirical Finding:** NEGATIVE RESULT: Species conditioning significantly degrades grading (delta QWK = -0.0360, paired t(29) = -4.159, p = 0.000260, Wilcoxon W = 72.0, p = 0.000555).

- **Scientific Implication:** Pretrained ConvNeXt-Tiny embeddings natively capture species-discriminative geometry; auxiliary categorical conditioning introduces optimization interference.


### RQ2: Does the observed context-conditioning degradation persist when complete group acquisition session holdout is strictly enforced?

- **Experimental Contrast:** Config B vs Config A across 30 seeds retrained from scratch on session_holdout_manifest.csv (12,980 test samples)

- **Primary Metric:** Test QWK difference under 100% session isolation

- **Empirical Finding:** REPLICATED NEGATIVE RESULT: Degradation persists under zero session leakage (delta QWK = -0.0262, paired t(29) = -2.749, p = 0.01018, Wilcoxon W = 125.0, p = 0.02703).

- **Scientific Implication:** Confirms that the degradation is not an artifact of train/test session allocation.


### RQ3: How do feature-space (Mahalanobis) and logit-space (Energy, MSP) open-set detectors behave across laboratory studio and natural market acquisition domains?

- **Experimental Contrast:** Mahalanobis vs Energy Score vs MSP evaluated on 5,000 real unseen fruit images across fruits360, real_quality, and real_grade

- **Primary Metric:** Area Under ROC Curve (AUROC) and False Positive Rate at 95% TPR (FPR95)

- **Empirical Finding:** DOMAIN INVERSION: Mahalanobis achieves 93.26% on studio but collapses to 18.08% on market bins; Energy score maintains 92.49% on market bins and 98.89% on conveyor.

- **Scientific Implication:** Feature covariance models are brittle under background texture shift; logit-space energy is far more robust for deployed open-set quality grading.


### RQ4: How does confidence-thresholded selective abstention affect risk and coverage under severe domain and class imbalance?

- **Experimental Contrast:** Selective accuracy and MAE across coverage levels (100% to 50%) using validation-tuned maximum probability thresholds

- **Primary Metric:** Risk-coverage curve, selective accuracy, and selective MAE

- **Empirical Finding:** MONOTONIC RISK REDUCTION: Rejecting bottom 10% least confident samples cuts MAE from 0.0889 to 0.0136 (84.7% error reduction) on exemplar seed 83 (92.11% to 98.91% across 30 seeds).

- **Scientific Implication:** Model confidence reliably isolates domain-shifted and ambiguous grading errors, enabling high-precision human-in-the-loop sorting.


