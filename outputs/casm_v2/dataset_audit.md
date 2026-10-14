# Dataset & Label Audit Report (Phase 4.2)

**Total Samples:** 56,036 across 5 Economic Species (*Apple, Banana, Mango, Orange, Tomato*)
**Acquisition Domains:** 4 (*fruits360, real_quality, real_severity, real_grade*)

## Split & Taxonomy Sample Distribution

| Severity Rank | Label Name | Train Split | Val Split | Test Split | Total | Ratio |
|---|---|---:|---:|---:|---:|---:|
| **0** | Healthy | 36,414 | 1,037 | 11,605 | 49,056 | 87.54% |
| **1** | Mild Defect | 846 | 494 | 88 | 1,428 | 2.55% |
| **2** | Moderate Defect | 327 | 196 | 0 | 523 | 0.93% |
| **3** | Severe Defect | 1,606 | 1,268 | 868 | 3,742 | 6.68% |
| **4** | Rotten | 716 | 418 | 153 | 1,287 | 2.30% |

## Species Distribution
| Fruit Species | Total Samples | Ratio |
|---|---:|---:|
| Apple | 24,962 | 44.55% |
| Banana | 6,141 | 10.96% |
| Mango | 1,842 | 3.29% |
| Orange | 6,927 | 12.36% |
| Tomato | 16,164 | 28.85% |

## Acquisition Domain Breakdown
| Domain Name | Description / Source | Samples | Ratio |
|---|---|---:|---:|
| `fruits360` | Clean synthetic background, pristine fruit baseline | 49,056 | 87.54% |
| `real_quality` | Real-world retail quality tiers (good, mixed, bad) | 3,742 | 6.68% |
| `real_severity` | Defect severity progression (fresh, mild, rotten) | 2,715 | 4.85% |
| `real_grade` | Export grading standard (grade_1, grade_2, grade_3) | 523 | 0.93% |

## Split Leakage & Session Isolation Audit
- **Train vs Val Path Overlap:** 0 images
- **Train vs Test Path Overlap:** 0 images
- **Val vs Test Path Overlap:** 0 images
- **Session/Directory Grouping:** Splitting performed at fruit/folder level preventing data leakage.

## Open-Set Unknown Species Audit
- **In-Distribution Species:** 5 species (*Apple, Banana, Mango, Orange, Tomato*)
- **Out-of-Distribution Unknown Species:** 5 species (*Cherry, Kiwi, Peach, Pineapple, Plum*) in `open-set-dataset`
- **Annotation Protocol Note:** OOD unknown species represent true open-set inputs. Severity labels on unknown fruits are evaluated for OOD rejection rather than assuming ground truth prior.
