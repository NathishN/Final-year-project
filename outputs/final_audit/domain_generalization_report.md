# Domain Generalization & Background-Shortcut Audit Report

**Date:** October 2026  
**Scope:** Dissection of Synthetic Turntable vs Real Market Acquisitions  

---

## 1. Domain Performance Breakdown

| Acquisition Domain | Sample Count | Pct of Test (%) | Config A Acc (%) | Config B Acc (%) | Config A MAE | Config A QWK | Label Variance |
|:---|---:|---:|---:|---:|---:|:---:|---:|
| **fruits360** | 10,974 | 85.51% | 99.88% | 99.74% | 0.0012 | UNDEFINED (Constant Label) | 0.0 |
| real_grade | 67 | 0.52% | 19.4% | 13.43% | 1.1642 | 0.2800 | 1.2667 |
| real_quality | 1,480 | 11.53% | 39.86% | 38.72% | 0.8135 | 0.5920 | 1.9714 |
| real_severity | 312 | 2.43% | 82.69% | 81.09% | 0.1731 | 0.8980 | 3.3461 |
| **REAL_COMBINED (split_real)** | 1,859 | 14.49% | 46.32% | 44.92% | 0.7187 | 0.7045 | 2.1922 |

---

## 2. Mathematical Resolution: Degenerate QWK on `fruits360`

In historical Phase 4.5 reports, `fruits360` showed `Accuracy = 99.88%`, `MAE = 0.0012`, but `QWK = 0.0000`. This led to confusion about whether agreement was perfect or completely broken.

### Mathematical Derivation:
Quadratic Weighted Kappa is defined as:
$$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}$$
where $O_{ij}$ is the observed confusion matrix and $E_{ij} = \frac{r_i c_j}{N}$ is the expected confusion under independent marginals.

- On `fruits360`, **100% of the 10,974 test samples have ground truth label $y = 0$** (Healthy).
- Consequently, the marginal ground-truth distribution is a degenerate delta function: $r_0 = N$ and $r_k = 0$ for $k > 0$.
- Therefore, the expected matrix $E_{0j} = c_j$ matches the observed marginal distribution exactly, causing $\sum w_{ij} O_{ij} = \sum w_{ij} E_{ij} = 0$.
- This yields an indeterminate form $\frac{0}{0}$. Standard software libraries (scikit-learn) return `0.0` by default.

**Audit Resolution:** QWK is **mathematically undefined** when true label variance is zero. Reporting QWK = 0.0 without explanation is misleading. The report properly marks this metric as **UNDEFINED (Zero Label Variance)**.

---

## 3. Background-Shortcut and Domain Confounding Findings

1. **Severe Accuracy Collapse:**  
   - Turntable studio domain (`fruits360`): **99.88% accuracy**
   - Real-world market bins (`split_real`): **46.32% accuracy** (MAE: 0.7187)
   - Real sorting station (`real_grade`): **19.40% accuracy** (MAE: 1.1642)

2. **Evidence of Background Shortcut Reliance:**  
   `fruits360` images feature clean, isolated fruits on white circular turntables with bright, omnidirectional lighting.    Real market images feature fruits embedded in corrugated cardboard, plastic crates, wooden slats, and natural shadows.    The deep ConvNeXt visual encoder extracts strong background texture and illumination representations that    fail to generalize zero-shot to commercial bin sorting.

3. **Proposed Leave-One-Domain-Out (LODO) Protocol:**  
   To measure true agricultural generalization, models must be evaluated on unseen domains rather than random i.i.d. splits.    We formalize the LODO protocol: Train on $\{D_1, D_2, D_3\}$, evaluate on held-out $D_4$.