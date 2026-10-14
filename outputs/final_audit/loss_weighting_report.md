# Class Imbalance Loss Weighting Audit Report

**Date:** October 2026  
**Partition Scope:** Calculated strictly on `split_v2` training partition ($N=39,790$)  

---

## 1. Class Distribution & Weighting Scheme Comparison

| Severity Grade | Clinical Name | Train Samples | Pct (%) | Unweighted | Raw Inverse | Effective Number (β=0.999) |
|:---:|:---|---:|---:|---:|---:|---:|
| 0 | Healthy | 36,414 | 91.52% | 1.0000 | 0.0171 | 0.4433 |
| 1 | Mild | 846 | 2.13% | 1.0000 | 0.7354 | 0.7762 |
| 2 | Moderate | 208 | 0.52% | 1.0000 | 2.9912 | 2.3594 |
| 3 | Severe | 1,606 | 4.04% | 1.0000 | 0.3874 | 0.5545 |
| 4 | Rotten | 716 | 1.8% | 1.0000 | 0.8689 | 0.8667 |

---

## 2. Weight Ratio Analysis and Instability Mechanism

- **Raw Inverse Frequency Weight Ratio (Max/Min):** **175.07x** (Grade 2 vs Grade 0)
- **Effective Number Weight Ratio (Max/Min):** **5.32x** (Grade 2 vs Grade 0)

> **Audit Mechanism:** In historical Phase 4.0, raw inverse frequency weighting assigned Grade 2 a 175x higher gradient penalty than Grade 0. Because Grade 0 comprised 91.5% of the data, the optimizer compensated by pushing ordinal cutpoints severely toward Grade 2, causing massive false-positive explosions on Seeds 19 and 31.  
> **Effective Number Solution (Cui et al., CVPR 2019):** By modeling the marginal information gain of additional samples with $\beta=0.999$, the weight ratio is smoothly bounded to **5.32x**, providing sufficient minority-class gradient without destabilizing threshold convergence.

---

## 3. Protocol Decision
Effective Number weighting ($eta=0.999$) is **FROZEN** as the mandatory loss strategy for all Phase 5.0 confirmatory evaluations.