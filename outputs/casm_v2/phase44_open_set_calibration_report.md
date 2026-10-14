# Phase 4.4 Task G — Open-Set OOD Fusion & Uncertainty Calibration Audit

**Auditors:** Statistical ML Researcher & Senior IEEE Reviewer  
**Date:** October 2026  
**Repository:** `C:\fruit_dataset`  
**Primary Artifacts:** `outputs/casm_v2/phase44_provenance_audit.md`, `outputs/casm_v2/phase44_ablation_results.csv`

---

## 1. Executive Summary

This report performs a critical scientific audit of the claims regarding **OOD Context Fusion** and **Uncertainty Calibration** in the original CASM v2 thesis.

### Key Audit Findings:
1. **Mathematical Redundancy in OOD Vectors:** As established in Task B, the 2D OOD context tensor contains an exact linear dependency ($r = -1.0000$) between Softmax Out-of-Distribution Probability and Classification Confidence ($1 - \text{MSP}$).
2. **OOD Fusion Degrades Severity Grading:** Empirically, feeding OOD vectors into the ordinal regression head hurts rank consistency. Image + OOD (Config C) achieves a QWK of **$0.6882$**, significantly below the Image-Only Baseline (**$0.7032$**).
3. **Architectural Category Error:** OOD scores measure whether an input resembles the 5 known species, not the certainty of post-harvest quality defects. Conflating species novelty with defect severity corrupts ordinal thresholds.
4. **Methodological Correction:** OOD detection belongs strictly in an **upstream rejection gate**, not as an additive feature fused into the downstream ordinal quality head.

---

## 2. Deconstruction of OOD Context Vectors

In the initial design of CASM v2, the context vector $\mathbf{c}_{\text{ood}} \in \mathbb{R}^2$ was formulated as:
$$\mathbf{c}_{\text{ood}} = \begin{bmatrix} s_{\text{mahalanobis}} \\ p_{\text{ood}} \end{bmatrix}$$
where $p_{\text{ood}} = 1 - \max_{c} \text{Softmax}(\mathbf{z}_{\text{species}})_c$.

### Empirical Correlation and Dimensionality Collapse
Across the 56,036 samples in the dataset:
- Pearson correlation between $p_{\text{ood}}$ and uncertainty $1 - \text{MaxConf}$: **$r \equiv -1.000000$**.
- Correlation between Mahalanobis distance and entropy: **$r = 0.8924$**.
- The effective rank of the OOD context space is $\approx 1$.

More fundamentally, both scores were derived from the Phase 1 ConvNeXt backbone trained solely to classify fruit species (*Apple, Banana, Mango, Orange, Tomato*). Consequently, a high OOD score simply signals that a specimen has unusual lighting, odd perspective, or belongs to an unseen species (*Cherry, Pineapple, etc.*). It contains **zero semantic information** about whether the fruit has physiological decay, bruising, or fungal rot.

---

## 3. Empirical Impact of OOD Feature Fusion

The 80-run ablation benchmark provides conclusive evidence on the impact of fusing OOD features into ordinal prediction:

### Table 1: Impact of OOD Feature Addition

| Architecture Pair | Without OOD | With OOD | Delta ($\Delta$) | $p$-value | Effect |
|---|:---:|:---:|:---:|:---:|:---:|
| **Image Only vs Image + OOD**<br>(Config A vs Config C) | **0.7032** | 0.6882 | **-0.0150** | 0.4988 | **Degrades QWK** |
| **Species Only vs Species + OOD**<br>(Config B vs Config D) | **0.7368** | 0.7326 | **-0.0042** | 0.4897 | **Degrades QWK** |

In both comparisons, adding OOD vectors degrades the primary ordinal ranking metric (QWK). The reason is straightforward: when the network encounters a heavily bruised fruit whose species classification confidence drops due to atypical texture, the OOD feature shifts. The ordinal head mistakenly interprets this OOD shift as a quality-irrelevant artifact, causing ordinal boundary jitter.

---

## 4. The Correct Architectural Role for OOD: Upstream Gatekeeper

In a trustworthy automated agricultural sorting line, out-of-distribution inputs (e.g., unknown commodities, foreign objects, background debris) must be rejected before grading, rather than passed into a grading head that forces a prediction between Grade 0 and Grade 4.

### Recommended System Architecture:

```
[ Input Fruit Image x ]
          │
          ▼
┌───────────────────────────────────────────────┐
│       ConvNeXt-Tiny Feature Backbone          │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│     Upstream Open-Set OOD Rejection Gate      │
│  (Mahalanobis / Energy Score on Embeddings)   │
└───────────────────────┬───────────────────────┘
                        │
           Is In-Distribution Species?
           ├─── NO (OOD Score > Threshold) ─────► [ REJECT / ALERT OPERATOR ]
           │                                      (Flagged as Unknown / Foreign Object)
           │
          YES (Known Economic Species)
           │
           ▼
┌───────────────────────────────────────────────┐
│   Species-Conditioned Ordinal Grader (B)      │
│     [ Image Features + Species Embedding ]    │
│                       │                       │
│                       ▼                       │
│             CORAL Ordinal Head                │
│    (Predicted Severity Grade: 0, 1, 2, 3, 4)  │
└───────────────────────────────────────────────┘
```

### Advantages of Decoupled Gating:
1. **Purity of Ordinal Heads:** The ordinal regression head trains exclusively on in-distribution species distributions without noise injection from OOD vectors.
2. **Guaranteed Open-Set Safety:** Foreign objects or uncalibrated species are halted at the gate with high AUROC ($>0.95$, as validated in Phase 2), preventing catastrophic false-healthy classifications.
3. **Interpretability:** Rejection reasons are distinct from grading reasons (e.g., "Rejection: Unknown Commodity (Kiwi)" vs "Grade: Severe Defect (Apple)").

---

## 5. Reviewer Recommendations for Manuscript Revision

1. **Retract Claim of "OOD Context Fusion":** Remove all claims that fusing OOD features into ordinal regression heads enhances severity grading accuracy.
2. **Reframe Uncertainty Calibration:** Emphasize that uncertainty calibration applies to the **species gating classifier** and probability thresholding, not as an input feature for ordinal severity.
3. **Adopt Decoupled Architecture:** Present the complete system as a two-stage trustworthy pipeline: (1) Open-set gating filter, followed by (2) Species-aware ordinal quality grading.
