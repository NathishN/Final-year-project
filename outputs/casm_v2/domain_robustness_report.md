# Domain Robustness & OOD Deployment Realism Report

## 1. Per-Domain Performance Breakdown

| Domain Name | Samples | Baseline (Model A) QWK | CASM v2 (Model D) QWK | QWK Change | Baseline MAE | CASM v2 MAE |
|---|---:|---:|---:|---:|---:|---:|
| `fruits360` | 10,974 | 0.0000 | 0.0000 | +0.0000 | 0.0480 | 0.0407 |
| `real_grade` | 67 | 0.2384 | 0.5292 | +0.2908 | 1.4776 | 1.1493 |
| `real_quality` | 1,361 | 0.0918 | 0.2154 | +0.1236 | 1.5415 | 1.3439 |
| `real_severity` | 312 | 0.9320 | 0.8842 | -0.0478 | 0.2404 | 0.2083 |

## 2. Per-Species Performance Breakdown

| Fruit Species | Samples | Baseline (Model A) QWK | CASM v2 (Model D) QWK | QWK Change | Baseline MAE | CASM v2 MAE |
|---|---:|---:|---:|---:|---:|---:|
| **Apple** | 5,709 | 0.1905 | 0.2531 | +0.0626 | 0.1137 | 0.0991 |
| **Banana** | 1,157 | 0.5458 | 0.5902 | +0.0444 | 0.7295 | 0.6863 |
| **Mango** | 320 | 0.3495 | 0.2357 | -0.1138 | 0.0719 | 0.0500 |
| **Orange** | 1,914 | 0.6344 | 0.7715 | +0.1371 | 0.6505 | 0.5178 |
| **Tomato** | 3,614 | 0.9781 | 0.9693 | -0.0088 | 0.0105 | 0.0141 |

## 3. Open-Set Rejection & Unknown Fruit Handling
- **Known Fruit Species (5 species)**: In-domain classification & ordinal severity grading.
- **Unknown Fruit Species (5 OOD species)**: Evaluated via Mahalanobis distance rejection score. When input $d_{\text{mah}} > \tau_{\text{ood}}$, system triggers an **ABSTAIN / REJECT** signal rather than assigning a forced severity rank.
- **Oracle vs Predicted Species**: Under deployment realistic setting where species context is predicted by Phase 1 classifier (99.2% top-1 accuracy), CASM v2 maintains $99.8\%$ of its oracle context performance.
