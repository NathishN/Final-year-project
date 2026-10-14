# Final Scientific Verdict & Publication Readiness Assessment

**Manuscript:** *Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation*  
**Date:** October 2026  
**Final Audited Outcome:** **OUTCOME C -- THE CORRECTED EXPERIMENT DEMONSTRATES NO MEANINGFUL BENEFIT FROM CONTEXT CONDITIONING**  

---

## Executive Scientific Synthesis

The comprehensive Phase 5.0 master audit and 30-seed confirmatory study have established a definitive, empirical resolution to the research questions posed in this project:

1. **Primary Finding on Multi-Modal Conditioning:**  
   When data leakage and threshold inversions are rigorously eliminated, **auxiliary species embeddings and OOD features do NOT improve    ordinal severity grading over pure visual ConvNeXt-Tiny representations**.    Instead, species conditioning significantly degrades Quadratic Weighted Kappa by $\Delta = -0.0360$ ($p = 0.000259$, $d_z = -0.7593$),    and OOD conditioning degrades QWK by $\Delta = -0.0700$ ($p < 10^{-7}$).    This is a methodologically valuable, reproducible negative finding.

2. **Primary Finding on Open-Set Recognition:**  
   The historical claim of 100% AUROC was an artifact of synthetic Gaussian sampling.    On 5,000 real images of unencountered fruit species, Mahalanobis Distance achieves **84.22% AUROC** (95% CI: [83.7%, 84.8%])    and Energy Score achieves **81.99% AUROC**, establishing a defensible empirical baseline.

3. **Primary Finding on Real-World Utility:**  
   The model exhibits severe domain shift: sorting accuracy collapses from **99.88%** on white-turntable laboratory images    to **46.32%** on real-world grocery and market-bin acquisitions (`split_real`).    Unqualified claims of 'industrial conveyor deployment readiness' are scientifically unsupported.

---

## Complete Claim-Evidence Summary

| Claim ID | Core Claim Statement | Primary Metric / Evidence | Statistical Test | Final Verdict |
|:---:|:---|:---|:---:|:---:|
| **CLAIM-01** | Species conditioning improves ordinal grading performance. | ΔQWK = -0.0360, 95% CI: [-0.0537, -0.0183] | Paired t-test t(29)=-4.16, p=0.000259; Wilcoxon p=0.000928 | **UNSUPPORTED (Statistically Significant Degradation)** |
| **CLAIM-02** | OOD distance conditioning improves ordinal grading. | ΔQWK = -0.0700, 95% CI: [-0.0902, -0.0498] | Paired t-test t(29)=-7.09, p=8.41e-08 | **UNSUPPORTED (Severe Degradation)** |
| **CLAIM-03** | Combining species and OOD conditioning improves grading. | ΔQWK = -0.0848, 95% CI: [-0.1008, -0.0687] | Paired t-test t(29)=-10.80, p=1.13e-11 | **UNSUPPORTED (Severe Degradation)** |
| **CLAIM-04** | The model generalizes zero-shot to real-world fruit images. | Real Accuracy = 46.32%, Real MAE = 0.7187 (vs 99.88% on fruits360) | Stratified cross-domain accuracy gap: -53.56% | **UNSUPPORTED (Severe Real-World Accuracy Collapse)** |
| **CLAIM-05** | The OOD detector recognizes unseen species rather than image-source differences. | Overall AUROC = 84.22% [83.7%, 84.8%]; Real-Quality AUROC = 18.08% | 1,000-resample bootstrap CI; Domain-stratified AUROC | **PARTIALLY SUPPORTED (Supported in-domain; Inverted on market bins)** |
| **CLAIM-06** | The model is calibrated (discrete probabilities match empirical accuracy). | ECE = 2.51%, Brier Score = 0.0887 | 10-bin Expected Calibration Error | **SUPPORTED (Low ECE, Well-Calibrated on Pooled Test)** |
| **CLAIM-07** | Selective abstention improves system sorting reliability. | Accuracy increases from 91.15% (100% cov) to 99.14% (90% cov), MAE: 0.1382 -> 0.0166 | Monotonic risk-coverage trade-off curve | **SUPPORTED (Abstention effectively filters high-risk errors)** |
| **CLAIM-08** | Background shortcuts explain domain performance differences. | Accuracy collapses from 99.88% to 19.40% on real_grade and 39.86% on real_quality | Stratified domain disparity analysis | **PARTIALLY SUPPORTED (Strong circumstantial evidence of background bias)** |
| **CLAIM-09** | The proposed architecture is parameter-efficient. | Head params = 626k-630k (2.2% of total 28.45M parameter pipeline) | Exact parameter summation | **SUPPORTED (Head is lightweight, though full pipeline requires 28.45M)** |
| **CLAIM-10** | The system is suitable for commercial automated fruit sorting. | Real accuracy = 46.32%, MAE = 0.7187 grades | Real-world sorting tolerance evaluation | **UNSUPPORTED (Premature claim; requires domain adaptation before commercial deployment)** |

---

## Definitive Publication Readiness Verdict

- **Outcome A (Superiority Supported):** REJECTED (Contradicted by 30-seed confirmatory evidence)
- **Outcome B (Inconclusive):** REJECTED (Difference is statistically significant with $p < 0.001$, not inconclusive)
- **Outcome C (No Benefit / Negative Finding):** **ACCEPTED AND ADOPTED**
- **Outcome D (Unresolved Integrity):** REJECTED (All data leakage and implementation defects fully resolved)

**Conclusion:** The manuscript is **READY FOR SUBMISSION** as a rigorous, honest empirical contribution and methodological audit.