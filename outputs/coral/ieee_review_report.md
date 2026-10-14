# IEEE Reviewer Report: Full Project Review (Phases 1-3)

**Paper**: Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using
Context-Aware Severity Modeling and Uncertainty Calibration

**Reviewer**: Associate Editor / Reviewer #2
**Venue**: IEEE Transactions on Instrumentation and Measurement (TIM) / IEEE Access
**Review**: Full technical review (Phases 1-3)

---

## Summary

This paper proposes a comprehensive framework for fruit quality grading
integrating: (1) closed-set multi-fruit classification using ConvNeXt-Tiny
with class-balanced focal loss, (2) open-set recognition via Mahalanobis
distance, and (3) rank-consistent ordinal regression using CORAL for
three-level quality grading. Phases 4-5 (severity modelling, uncertainty
calibration) are planned but incomplete.

The engineering execution is thorough, but several scientific concerns must
be addressed before this work meets the IEEE publication bar.
## Review Scores

            Dimension Score                                                              Comment
              Novelty  7/10 CORAL + Mahalanobis OOD combination in agricultural grading is novel
    Technical Quality  8/10       Implementation rigorous; two-stage training rationale is sound
  Experimental Design  6/10                Tomato test split gap and small dataset weaken claims
      Reproducibility  9/10             Session-level splits, seeds, checkpoints well-documented
Publication Readiness  6/10                      Phases 4-5 incomplete; statistical tests needed

## Major Strengths

**S1 -- Dataset Curation Rigor:**
The multi-domain dataset construction with session-level leakage prevention
is a genuine methodological contribution. Zero session contamination across
56,038 images is verifiable and reproducible. This is significantly stronger
than image-level random splits common in prior fruit quality work.

**S2 -- Ordinal Loss Principled Design:**
CORAL's shared-weight formulation correctly encodes the monotonic structure
of agricultural quality grades. The loss directly minimizes violations of
P(y>0|x) >= P(y>1|x), unlike CE which is ordinally agnostic.

**S3 -- Open-Set Integration:**
Combining OOD rejection (Mahalanobis) with ordinal grading is uncommon in
agricultural AI literature. The unified trustworthy system concept is compelling.

**S4 -- Computational Efficiency:**
ConvNeXt-Tiny (28M parameters) achieves strong performance with significantly
lower FLOPs than ViT-based alternatives. RTX 3050 constraint demonstrates
practical deployability in resource-constrained settings.

**S5 -- Two-Stage Transfer Learning:**
The head-only pre-training strategy is well-motivated for small datasets and
correctly prevents catastrophic forgetting during backbone fine-tuning.
## Major Weaknesses (Required Revisions)

**W1 [Critical] -- Incomplete System:**
The title promises "Context-Aware Severity Modelling and Uncertainty Calibration"
but Phases 4-5 are not implemented. A paper cannot be submitted without validating
all title contributions experimentally.

Required revision: Complete Phases 4-5 or modify title/claims to match current implementation.

**W2 [Critical] -- Tomato Evaluation Gap:**
Tomato comprises 31.3% of real_grade samples yet has ZERO test samples due to
session-level splitting. Per-species results for Tomato are evaluation-set only
(data the model was optimised on). Species-level generalisation claims are
scientifically invalid for Tomato.

Required revision: Apply 5-fold CV guaranteeing each species appears in at
least one test fold. Report per-species metrics with fold-averaged estimates.

**W3 [Critical] -- Missing Statistical Testing:**
The ablation and multi-seed experiments must include:
- Paired t-test with p-values for CORAL vs CE
- Effect size (Cohen's d)
- 95% confidence intervals
- Bonferroni correction for multiple metrics

Without these, "CORAL is better than CE" is not a scientifically defensible claim.

**W4 [Significant] -- AUROC=1.000 for OOD is Unconvincing:**
Perfect OOD detection raises the question of whether the evaluation is sufficiently
challenging. Testing with a completely different fruit species (e.g., Orange) as
the OOD distribution is required.

Required revision: Report OOD AUROC with held-out species as the negative class.

**W5 [Moderate] -- No Comparison to SOTA Ordinal Methods:**
Missing baselines: CORN (Shi et al. 2022), Ordinal Encoding, Cumulative Link
Models, and recent ordinal transformers.

Required revision: Add at least 2 ordinal regression baselines.
## Minor Revisions

m1: Report inter-rater reliability (Cohen's kappa) for grade annotations.
m2: Provide GradCAM visualizations confirming model attends to fruit surface, not background.
m3: Table IV (baseline comparison) requires absolute accuracy/F1 numbers, not "Best/--".
m4: Clarify whether Phase 1 accuracy includes OOD filtering or is closed-set only.
m5: Report the best epoch number and early stopping behavior explicitly in the paper.
## Recommendation

## Decision: Major Revision

The engineering quality and reproducibility are strong. The core scientific
claims are plausible but not yet proven due to:

1. Incomplete system (Phases 4-5 missing)
2. Invalid Tomato test evaluation (zero test samples)
3. Missing statistical significance testing
4. Unconvincing OOD evaluation (AUROC=1.000)
5. Missing ordinal regression baselines

Upon addressing W1-W5, this paper has strong potential for acceptance in:
- **IEEE Access** (good breadth, practical contribution)
- **IEEE Transactions on Instrumentation and Measurement** (strong dataset + system focus)

For **IEEE TNNLS** or **Pattern Recognition Letters**, the severity modelling
contribution (Phase 4) must be fully validated and compared to prior work.

**Overall Rating: 5/10 (Borderline) -- Major Revision Required**
