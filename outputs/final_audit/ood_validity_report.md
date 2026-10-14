# Real-Image Open-Set Recognition & Domain Confounding Audit Report

**Date:** October 2026  
**Unknown Fruit Sample Size:** 5,000 real images across 5 unencountered species  
**Known Fruit Test Set:** 12,833 images across 5 trained species (split_v2)  

---

## 1. Resolution of Historical Synthetic OOD Defect

> **Historical Defect (Phase 2.0 / 4.5 Audit):** The previously reported 100.0% AUROC was derived from synthetic Gaussian feature values artificially sampled in latent space. It did not reflect actual open-set images.  
> **Phase 5.0 Correction:** Extracted real 768-D ConvNeXt-Tiny embeddings on 5,000 real images from `open-set-dataset/` (Cherry, Kiwi, Peach, Pineapple, Plum).

---

## 2. Overall Real-Image OOD Benchmark Results

| Detector Method | AUROC (%) | 95% Bootstrap CI (1,000 resamples) | AUPR (OOD=1) (%) | FPR @ 95% TPR (%) |
|:---|---:|:---:|---:|---:|
| **Mahalanobis** | **84.22%** | [83.7%, 84.8%] | 53.28% | 38.39% |
| **Energy** | **81.99%** | [81.4%, 82.6%] | 49.96% | 41.38% |
| **MSP** | **74.26%** | [73.6%, 75.0%] | 43.16% | 47.85% |

---

## 3. Per-Unknown-Species Performance Breakdown

| Unknown Fruit Species | Evaluation N | Mahalanobis AUROC | Energy AUROC | MSP AUROC |
|:---|---:|---:|---:|---:|
| **Cherry** | 1,000 | 76.99% | 81.02% | 69.06% |
| **Kiwi** | 1,000 | 84.77% | 82.84% | 79.12% |
| **Peach** | 1,000 | 78.28% | 74.21% | 71.98% |
| **Pineapple** | 1,000 | 93.00% | 84.23% | 74.94% |
| **Plum** | 1,000 | 88.08% | 87.64% | 76.21% |

---

## 4. Crucial Discovery: Domain Confounding Analysis

| In-Distribution Domain | Sample N | In-Dist Characteristic | Mahalanobis AUROC | Interpretation |
|:---|---:|:---|---:|:---|
| `fruits360` | 10,974 | Clean white turntable background | **93.26%** | Excellent species discrimination within matched studio domain |
| `real_severity` | 312 | Field photography | **94.10%** | Robust separation against field fruit images |
| `real_quality` | 1,480 | Wooden crates & market bins | **18.08%** | **SEVERE DOMAIN CONVECTION**: Bin background misclassified as OOD |
| `real_grade` | 67 | Sorting station conveyor | **18.81%** | **SEVERE DOMAIN CONVECTION**: Lighting shifts confound detector |

### Scientific Significance of Confounding:
1. **Near-Zero Real-Market Open-Set Transfer:** When the in-distribution test samples come from market bins (`real_quality`), the Mahalanobis detector exhibits an inverted AUROC of 18.08%. The detector's feature representations are dominated by background textures, shadows, and crate materials rather than intrinsic fruit morphological features.
2. **Publication Guideline:** The manuscript MUST report both the pooled 84.22% AUROC and explicitly disclose the 18.08% failure mode on market-bin images. Claiming 'generalized industrial open-set sorting' without this disclosure is scientifically invalid.