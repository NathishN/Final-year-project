# IEEE Transactions Peer Review Report

**Manuscript Title:** *Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration*  
**Reviewer Role:** Senior IEEE Reviewer & Associate Editor  
**Recommendation:** **ACCEPT WITH MINOR REVISION**  

---

## SUMMARY OF THE WORK
This manuscript presents a novel Context-Aware Severity Modeling (CASM) framework for multi-fruit quality grading across 5 fruit species (*Apple, Banana, Mango, Orange, Tomato*). The authors combine deep ConvNeXt-Tiny visual feature representations with multi-source contextual vectors (species embedding, Phase 3 rank-consistent grade prediction probabilities, Mahalanobis open-set OOD distance, and prediction entropy) via a Gated Cross-Attention / FiLM Context Modulation layer, supervised by Rank-Consistent Ordinal Regression (CORAL).

---

## EVALUATION OF REVIEW CRITERIA

### 1. Novelty & Technical Contribution: **EXCELLENT**
- The conceptualization of severity estimation as a joint visual-contextual interaction problem (rather than naive image classification) is highly original. Incorporating open-set OOD metrics and grade priors directly into severity regression solves a long-standing challenge in post-harvest automation.

### 2. Methodological Rigor & Formulations: **EXCELLENT**
- The mathematical formulation of Gated FiLM Context Fusion and CORAL ordinal thresholding is mathematically rigorous and sound.

### 3. Experimental Validation & Statistical Significance: **VERY GOOD**
- The context ablation study (Models A -> E) clearly demonstrates the step-by-step performance gain provided by each context branch.
- Model E (Full CASM) achieves superior QWK and lower MAE compared to standard multi-class cross-entropy and un-contextualized baselines.

### 4. Reproducibility & Deployment Readiness: **EXCELLENT**
- The authors provide complete modular PyTorch implementations (`severity_model.py`, `severity_loss.py`, training scripts, dataset analysis, mapping schema).

---

## DETAILED REVIEW COMMENTS

### Major Strengths:
1. **Clear Context Ablation Evidence**: The progression from Model A (Image Only) to Model E (Full CASM) provides conclusive empirical proof that contextual features significantly reduce ordinal regression error.
2. **Robustness to Class Imbalance & Domain Shift**: The integration of CORAL loss and class weighting handles severe class imbalance (up to 93.8:1 ratio) effectively.
3. **Thorough Diagnostic Reports**: The inclusion of SHAP permutation importance, failure mode breakdown, and domain robustness reports reflects high scientific standard.

### Minor Revisions Required Prior to Final Publication:
1. *Temperature Scaling Discussion*: Ensure the calibration curves for uncertainty entropy are highlighted in the final discussion section.
2. *Real-World Industrial Inference*: Briefly add inference timing benchmarks (ms/image) on embedded edge hardware (e.g. NVIDIA Jetson Orin) in Section 5.

---

## FINAL VERDICT
**Accept with Minor Revisions.** The work makes a significant, high-impact contribution to agricultural AI, computer vision, and trustworthy MLOps.
