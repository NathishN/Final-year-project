# Forensic OOD Domain Mapping Reconciliation & Failure Mode Audit

**Audit Timestamp:** 2026-10-14 10:19:33  

**In-Distribution Test Size:** 12,833 samples  

**Out-of-Distribution Test Size:** 5,000 samples (Cherry, Kiwi, Peach, Pineapple, Plum)  

**Backbone:** ConvNeXt-Tiny (outputs/checkpoints/best_model.pth) (768-D)

---

## 1. Pooled Open-Set Detector Performance

| Detector | AUROC | AUPR (OOD) | FPR @ 95% TPR | 95% Confidence Interval |
|---|---:|---:|---:|:---:|
| **Mahalanobis** | **84.22%** | 53.28% | 38.39% | [83.7%, 84.8%] |
| **Energy** | **81.99%** | 49.96% | 41.38% | [81.3%, 82.6%] |
| **MSP** | **74.26%** | 43.16% | 47.85% | [73.6%, 74.9%] |


## 2. In-Distribution Domain Breakdown & The Inversion Phenomenon

Evaluating in-distribution samples against out-of-distribution samples stratified by the origin acquisition domain of the ID images uncovers severe domain confounding:

| In-Distribution Subset | Test Sample Count | Mahalanobis AUROC | Energy AUROC | MSP AUROC | Primary Visual Setting |
|---|---:|---:|---:|---:|---|
| `fruits360` | 10,974 | **93.26%** | **79.99%** | **75.21%** | Clean white studio turntable |
| `real_grade` | 67 | **18.81%** | **98.89%** | **96.57%** | Industrial conveyor / packing station |
| `real_quality` | 1,480 | **18.08%** | **92.49%** | **61.74%** | Grocery market bins & natural retail |
| `real_severity` | 312 | **94.10%** | **98.67%** | **95.30%** | Field / disease severity photography |


### Key Audit Observations on Domain Inversion:

1. **Mahalanobis Failure on Real Domains:** Mahalanobis distance achieves an outstanding **93.26% AUROC** when separating studio images (`fruits360`) from unseen fruits, but drastically collapses to **18.08% AUROC** on `real_quality` and **18.81% AUROC** on `real_grade`. An AUROC < 50% indicates that in-distribution market images are assigned higher Mahalanobis distance from the class centroids than out-of-distribution fruit images.

2. **Energy Score Robustness:** Energy Score exhibits the opposite pattern: it achieves **92.49% AUROC** on `real_quality` and **98.89% AUROC** on `real_grade`, while achieving 79.99% on `fruits360`. Logit-space pooling filters high-dimensional feature covariance shifts, maintaining robust discrimination in natural settings.

3. **MSP Conservatism:** Maximum Softmax Probability (MSP) shows intermediate behavior (75.21% on studio, 61.74% on market bins, 96.57% on conveyor), suffering from overconfident false positives on out-of-distribution samples.


## 3. Per-OOD Species Breakdown

| Unseen Fruit Species | OOD Sample Count | Mahalanobis AUROC | Energy AUROC | MSP AUROC |
|---|---:|---:|---:|---:|
| **Cherry** | 1,000 | 76.99% | 81.02% | 69.06% |
| **Kiwi** | 1,000 | 84.77% | 82.84% | 79.12% |
| **Peach** | 1,000 | 78.28% | 74.21% | 71.98% |
| **Pineapple** | 1,000 | 93.00% | 84.23% | 74.94% |
| **Plum** | 1,000 | 88.08% | 87.64% | 76.21% |


## 4. Formal Causal Attribution Assessment

**Attribution Status:** `UNRESOLVED`  


> **Scientific Auditor Statement:**  

> Photographic datasets lack pixel-level segmentation masks separating foreground fruit from background surfaces. While visual background differences (studio turntable white background vs natural grocery market bins) strongly correlate with feature distribution distortion, definitive causal attribution cannot be mathematically proven without intervention experiments or synthetic background swaps.


**Structural Mechanism Analysis:**  

Feature-space Mahalanobis distance relies on a class-conditional Gaussian prior in the 768-D ConvNeXt embedding space. Because 85.5% of the training distribution is studio white-background, the empirical covariance matrix Σ reflects studio background textures. When natural market bins (real_quality) are presented, background clutter shifts the feature vector far outside the studio Gaussian clusters, causing high Mahalanobis distance even for in-distribution fruits, inverting the detector AUROC to 18.08%. In contrast, logit-space Energy Score aggregates over class log-sum-exp logits, filtering out domain-specific visual texture shifts and achieving 92.49% AUROC on real_quality.


### Manuscript Presentation Policy:

- The manuscript MUST NOT claim that background texture is conclusively proven to cause the Mahalanobis collapse.

- The manuscript MUST present the domain-stratified results transparently as a domain confounding phenomenon, highlighting Energy Score as the recommended detector for deployed open-set quality grading.

