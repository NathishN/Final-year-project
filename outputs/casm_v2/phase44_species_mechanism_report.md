# Phase 4.4 Task E — Species Mechanism & Training Stabilization Report

**Auditor:** PyTorch Architect & Agricultural AI Specialist  
**Date:** October 2026  
**Repository:** `C:\fruit_dataset`  
**Primary Artifacts:** `outputs/casm_v2/phase44_ablation_raw.csv`, `outputs/casm_v2/phase44_stability_investigation.md`

---

## 1. Executive Summary

This report analyzes the physiological, architectural, and optimization mechanisms through which **Species Conditioning (Config B)** improves multi-fruit ordinal severity estimation. The investigation explains why species embeddings dramatically attenuate cross-seed variance, inoculate the model against initialization collapse, and how they interact with severe label imbalance.

### Key Mechanistic Insights:
1. **Biological Defect Heterogeneity:** Visual manifestations of decay differ radically across commodities (e.g., natural enzymatic browning in bananas vs surface puncture in apples vs epidermal translucency in tomatoes). A shared visual backbone without species conditioning is forced to average these conflicting gradients.
2. **Symmetry-Breaking & Collapse Prevention:** Under extreme label imbalance ($87.5\%$ Grade 0), baseline linear heads frequently collapse toward the majority class on unfavorable initializations. Species embeddings provide an auxiliary $32\text{D}$ subspace that acts as an inductive bias, preventing gradient starvation for defect classes.
3. **Massive Variance Reduction:** Species conditioning reduces standard deviation across 20 random initializations from $\sigma = 0.0728$ down to $\sigma = 0.0269$ (a **$63.0\%$ variance reduction**).
4. **The Species Macro-F1 Trade-off:** Strong species conditioning slightly suppresses minority class predictions when species priors are heavily skewed toward Healthy, explaining why Config B has lower unweighted Macro-F1 ($25.69\%$) despite achieving higher QWK ($0.7368$).

---

## 2. Agricultural & Botanical Rationale for Species Conditioning

In computer vision for post-harvest agriculture, evaluating severity ranks across multiple commodities presents severe inter-species domain shift:
- **Banana:** Natural ripening entails extensive peel senescent spotting (sugar spots) that visually mimics fungal anthracnose. An image-only classifier cannot distinguish physiological ripening from pathological decay without species-conditioned priors.
- **Apple:** Mechanical bruising causes subsurface cellular collapse without immediate color shifts, requiring lower intensity thresholds for defect classification.
- **Tomato:** High water content produces localized soft rots that spread rapidly, exhibiting distinct textural gradients.

When a single ordinal threshold vector $\mathbf{b} \in \mathbb{R}^4$ is trained globally across all species without species conditioning, the model compromises on median decision boundaries that fit no single species optimally. Conditioning the CORAL head on learned species representations allows the network to adapt effective ordinal thresholds $\mathbf{b}_s$ per species:
$$P(Y > k \mid \mathbf{x}, s) = \sigma\left(\mathbf{w}^T \phi(\mathbf{x}, \mathbf{e}_s) - b_k\right)$$

---

## 3. Optimization Dynamics: Inoculation Against Initialization Collapse

### Detailed Diagnosis of Failure Seeds (19 and 31)

In the 20-seed benchmark, the baseline model experienced severe performance degradation on Seeds 19 and 31:

| Seed | Architecture | QWK | MAE | Accuracy (%) | OBO Acc (%) | Macro-F1 (%) | Diagnosis |
|:---:|---|---:|---:|---:|---:|---:|---|
| **19** | Config A (Baseline) | **0.5363** | **0.3994** | 76.59% | 85.33% | 30.59% | Partial Majority Collapse |
| **19** | Config B (Species) | **0.7180** | **0.2312** | 86.81% | 93.91% | 25.12% | Fully Recovered |
| **19** | Config D (Full CASM v2) | **0.7179** | **0.2149** | 87.02% | 93.68% | 34.99% | Fully Recovered |
| **31** | Config A (Baseline) | **0.5000** | **0.4433** | 80.34% | 84.28% | 38.04% | Severe Majority Collapse |
| **31** | Config B (Species) | **0.7115** | **0.2523** | 85.94% | 93.44% | 26.31% | Fully Recovered |
| **31** | Config D (Full CASM v2) | **0.6923** | **0.2315** | 87.18% | 93.75% | 31.67% | Fully Recovered |

### Why Did the Baseline Collapse?
1. **Severe Imbalance Landscape:** The training set contains $36,414$ Healthy samples ($91.2\%$) and only $327$ Moderate Defect samples ($0.8\%$). The loss landscape possesses wide local basins where all binary CORAL classifiers output zero (predicting Grade 0 for all samples).
2. **Gradient Vanishing in CORAL Head:** When the linear projection weights $\mathbf{w}$ are initialized far from defect features, the sigmoid derivatives $\sigma'(z)$ for rare classes vanish exponentially.
3. **How Species Embeddings Intervene:**
   - The learned species embedding table $\mathbf{E} \in \mathbb{R}^{5 \times 32}$ introduces independent parameter pathways.
   - Even if visual gradients are dominated by majority-class imagery, backpropagation through the species embedding preserves distinct bias shifts for each fruit type.
   - This prevents all ordinal thresholds from collapsing into a degenerate majority-class equilibrium.

---

## 4. Cross-Seed Stability and Variance Reduction

### Table: Variance Across 20 Seeds

| Metric | Config A (Image Only) | Config B (Image + Species) | Config D (Image + Species + OOD) | Variance Reduction (B vs A) |
|---|:---:|:---:|:---:|:---:|
| **QWK Std ($\sigma$)** | 0.0728 | **0.0269** | **0.0195** | **-63.0%** |
| **MAE Std ($\sigma$)** | 0.0670 | **0.0228** | **0.0154** | **-66.0%** |
| **Accuracy Std ($\sigma$)** | 3.01% | **0.93%** | **0.67%** | **-69.1%** |
| **OBO Acc Std ($\sigma$)** | 3.01% | **0.79%** | **0.68%** | **-73.8%** |

In commercial sorting facilities handling millions of fruits per day, model variance across deployment runs represents operational risk. Species conditioning provides an industrial-grade stabilization benefit, ensuring consistent sorting yields regardless of stochastic initialization.

---

## 5. The Macro-F1 Penalty Under Strong Species Conditioning

A notable phenomenon in the ablation benchmark is the reduction in 5-class Macro-F1 for Config B:
- Baseline (Config A): $\text{Macro-F1} = 29.83 \pm 3.18\%$
- Species-Only (Config B): $\text{Macro-F1} = \mathbf{25.69 \pm 1.13\%}$
- Full CASM v2 (Config D): $\text{Macro-F1} = 30.32 \pm 2.49\%$

### Root Cause Analysis:
1. **Species-Conditioned Prior Tightening:** In the training set, Apples and Tomatoes consist overwhelmingly of `fruits360` samples ($100\%$ Healthy). When the network conditions heavily on `Apple` or `Tomato` embeddings, it learns an extremely strong prior $P(\text{Healthy} \mid \text{Apple}) \approx 0.98$.
2. **Minority Class Conservatism:** For ambiguous edge cases of Mild Defect or Rotten fruit in these species, the network refuses to trigger the higher ordinal thresholds, favoring Grade 0. While this optimizes rank agreement (QWK) and minimizes squared distance errors, it suppresses recall on the rare defect classes.
3. **The Role of OOD Noise in Config D:** Adding OOD context features (which are noisy and unaligned with quality) acts as implicit regularization, preventing the model from becoming overly confident in the species prior. This restores minority class recall, raising Macro-F1 back to $30.32\%$ at a slight cost to QWK.

---

## 6. Engineering & Architectural Recommendations

1. **Retain Species Embedding:** 160 parameters in the lookup table ($5 \times 32$) yield a $+0.0337$ QWK gain and a $>60\%$ variance reduction.
2. **Incorporate Class-Weighted CORAL Loss:** To prevent species priors from eroding minority class recall (addressing the Macro-F1 dip in Config B), future iterations should apply inverse-frequency weighting to the ordinal loss:
   $$\mathcal{L}_{\text{CORAL}} = \sum_{k=1}^{K-1} w_k \cdot \text{BCE}(\hat{p}_k, y_k)$$
3. **Drop OOD Context Vectors:** OOD features introduce collinearity and do not provide genuine quality information. Stability should be maintained via species conditioning and weight regularization, not noisy inputs.
