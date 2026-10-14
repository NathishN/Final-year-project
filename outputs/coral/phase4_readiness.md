# Task 8: Phase 4 Readiness Analysis

_Context-Aware Severity Modelling_

---

## 8.1 Current State Assessment

                    Criterion      Status                                           Evidence
    Feature quality (Phase 1)       Ready ConvNeXt-Tiny features extracted; AUROC=1.0 on OOD
Grade label quality (Phase 3) Conditional         1,602 samples; label noise risk in grade_3
  Severity label availability     Partial         real_severity domain (2,602 images) exists
      OOD rejection framework       Ready                      Mahalanobis fully implemented
 Ordinal regression (Phase 3)     Running                         CORAL training in progress
         Calibration baseline Not started                             Phase 5 required first

## 8.2 Phase 4 Architecture Recommendation

**Phase 4: Context-Aware Severity Modelling**

Goal: Given a fruit image, predict both the GRADE (1/2/3) AND the SEVERITY
of each defect attribute (bruising, discolouration, rot) using contextual
features from neighboring spatial regions.

Proposed Architecture:

    Input (224x224)
        -> ConvNeXt-Tiny (frozen from Phase 3 best checkpoint)
        -> Feature Map: F in R^{7x7x768}  (before global pool)
        -> Context Attention Module (CAM)
               Multi-head self-attention across spatial positions
               Captures WHERE the defect is and HOW SEVERE
        -> Global Context Vector: c in R^{768}
        -> Local Severity Head: MLP -> [bruising, discolouration, rot] in [0,1]^3
        -> Grade Prediction Head: CoralHead (reused from Phase 3)
        -> Joint Output: (grade, severity_vector)

Loss Function:

    L = lambda_g * L_CORAL + lambda_s * L_severity

where L_severity = BCE(s_hat, s*) on continuous severity scores,
and lambda_g = 1.0, lambda_s = 0.5 initially.

Key Design Choices:
1. Context attention: Preserves spatial defect localization, unlike global pooling.
2. Shared backbone: Reuse Phase 3 checkpoint -- no retraining from scratch.
3. Multi-task learning: Grade + severity trained jointly to regularize each other.
4. Dataset: Use real_severity (2,602 images) + real_grade (1,602 images) jointly.
## 8.3 Phase 4 Training Strategy

Stage 1 (3 epochs): Freeze backbone; train Context Attention Module + Severity Head only.

Stage 2 (15 epochs): Unfreeze last 2 ConvNeXt stages; joint fine-tune with multi-task loss.

Evaluation Metrics:
- Grade: MAE, QWK (same as Phase 3)
- Severity: Spearman correlation per attribute, Mean Absolute Error
- Joint: Grade QWK with severity-aware weighting

Expected Timeline: 2-3 days implementation, 4-6 hours training on RTX 3050.
## 8.4 Phase 4 Go / No-Go Criteria

               Criterion  Target                 Current Decision
        CORAL QWK > 0.50  > 0.50                  0.0000     HOLD
        CORAL MAE < 0.50  < 0.50                999.0000     HOLD
   Critical errors < 10%   < 10% See failure_analysis.md  Pending
Tomato eval gap resolved CV done    0 fixed test samples     HOLD


**Recommendation:**
If CORAL QWK > 0.50 and MAE < 0.50, proceed to Phase 4 with the context
attention architecture above. If not, invest in dataset expansion first
(target: 5,000+ real_grade images across all 3 species with balanced sessions).
