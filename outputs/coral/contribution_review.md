# Task 7: Research Contribution Validation

_IEEE Reviewer Perspective_

---

## 7.1 Contribution Scoring Matrix

                    Contribution Novelty Technical Depth Scientific Rigor Publication Strength                                                             Notes
      Open-Set Recognition (OOD)    7/10            8/10             7/10                 8/10    Mahalanobis is established; applying to fruit quality is novel
        CORAL Ordinal Regression    6/10            8/10             7/10                 7/10 CORAL (2020) is known; first application to fruit quality grading
Context-Aware Severity Modelling    8/10            7/10             6/10                 8/10         Phase 4 -- not yet implemented; highest novelty potential
         Uncertainty Calibration    7/10            7/10             6/10                 7/10            Phase 5 -- Temperature Scaling; standard but important
         Multi-Domain Robustness    8/10            9/10             8/10                 9/10  4-domain dataset with session-level leakage prevention is strong

## 7.2 Major Strengths

1. **Multi-domain dataset construction**: Merging fruits360, real_quality,
   real_severity, and real_grade with zero session leakage is a genuine
   engineering contribution. Session-level splitting is statistically more
   rigorous than image-level random splitting.

2. **Principled OOD integration**: Mahalanobis distance in ConvNeXt feature
   space provides a theoretically grounded uncertainty signal for open-set
   rejection. No additional training overhead required.

3. **Ordinal loss design**: CORAL's shared-weight formulation enforces monotonic
   rank consistency, which is semantically correct for agricultural quality grading.

4. **Complete ML pipeline**: The project covers the full stack: data -> features
   -> classification -> OOD -> ordinal -> severity -> calibration. This breadth
   is rare in agricultural AI literature.

5. **Reproducibility**: Session-level splits, fixed seeds, checkpoint saving, and
   multi-seed evaluation demonstrate strong reproducibility commitment.
## 7.3 Major Weaknesses

1. **Small real-domain dataset (N=1,602)**: Core contributions rest on 1,602
   real-world images. Insufficient for IEEE Transactions without CV + multi-seed.

2. **Tomato evaluation gap**: All 3 Tomato sessions are in train/val. Per-species
   results for Tomato are not derived from a genuine held-out test set.

3. **No inter-rater reliability**: Agricultural grade labels have inherent
   subjectivity. Without reporting kappa_inter-rater, label quality is unknown.

4. **AUROC=1.000 for OOD is suspicious**: May indicate evaluation is not
   sufficiently distribution-shifted. Held-out species OOD is required.

5. **Phases 4-5 incomplete**: Title promises Context-Aware Severity Modelling
   and Uncertainty Calibration -- both are not yet implemented.
## 7.4 Critical Risks

                                          Risk   Prob   Impact                                  Mitigation
CORAL shows no significant improvement over CE Medium     High                 Multi-seed CI; focus on QWK
     OOD AUROC=1.0 is train-test contamination Medium Critical             Test on genuinely novel species
    Label noise in grade_3 causes floor effect   High   Medium Manual grade_3 misclassification inspection
    Dataset too small for Transactions journal   High     High          Expand real_grade to 5,000+ images
       Phase 4/5 not novel enough individually Medium   Medium         Frame as unified trustworthy system

