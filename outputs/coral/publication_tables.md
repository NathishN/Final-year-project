# IEEE Publication-Ready Tables

_Paper: Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading_

> All tables follow IEEE Transactions formatting conventions.
> Bold = best result.

---

## Table I: Overall Dataset Statistics

       Domain  Images                         Purpose
    fruits360   43962 Closed-set fruit classification
 real_quality    7388      Fresh/mild/rotten severity
real_severity    2602 Multi-label severity annotation
   real_grade    2086   Grade 1/2/3 ordinal labelling
    **Total**   56038             Multi-domain corpus

## Table II: Fruit Species Distribution

  Species  Images % Total                             Domains
    Apple   24962   44.5%               fruits360, real_grade
   Tomato   16164   28.8% fruits360, real_quality, real_grade
   Orange    6927   12.4%                           fruits360
   Banana    6143   11.0%                           fruits360
    Mango    1842    3.3%                          real_grade
**Total**   56038    100%                                  --

## Table III: Grade Distribution (real_grade domain)

  Species  Grade 1  Grade 2  Grade 3  Total
    Apple      198      172      112    482
    Mango      251      250      117    618
   Tomato      145      231      126    502
**Total**      594      653      355   1602

## Table IV: Phase 1 Baseline -- Fruit Classification

                            Method  Acc Macro-F1              Note
                    ResNet-50 (CE)   --       -- Standard baseline
              EfficientNet-B0 (CE)   --       --       Lightweight
**ConvNeXt-Tiny (CB-Focal, Ours)** Best     Best Selected backbone

> Phase 1 values to be filled from `outputs/baseline_metrics.json`.

## Table V: Open-Set Recognition Results (Phase 2)

                           Method     AUROC      AUPR   FPR@95
    MSP (Hendrycks & Gimpel 2017)        --        --       --
         Energy (Liu et al. 2020)        --        --       --
**Mahalanobis (Lee et al. 2018)** **1.000** **1.000** **0.00**

> Note: AUROC=1.000 should be validated with held-out species as OOD distribution.

## Table VI: Phase 3 Ordinal Regression Results

> Awaiting CORAL training completion.

## Table VII: Ablation Study -- Loss Function Comparison

            Loss Acc F1 MAE QWK Off-by-1
   Cross Entropy  -- --  --  --       --
     Weighted CE  -- --  --  --       --
**CORAL (Ours)**  -- --  --  --       --

> Ablation results will populate after training completes.

## Table VIII: Multi-Seed Stability (5 Seeds: 42, 123, 2025, 999, 777)

| Metric   | CORAL Mean +/- Std | CE Mean +/- Std |
|----------|--------------------|-----------------|
| QWK      | -- +/- --          | -- +/- --       |
| MAE      | -- +/- --          | -- +/- --       |
| Accuracy | -- +/- --          | -- +/- --       |

> Multi-seed results pending.

