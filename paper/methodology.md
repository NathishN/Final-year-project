# Methodology

## 1. Problem Formulation and Ordinal Framework

Let $\mathcal{X} \subset \mathbb{R}^{H \times W \times 3}$ denote the input image space, and let $y \in \mathcal{Y} = \{0, 1, \dots, K-1\}$ denote the ordinal severity grade, where $K=5$ corresponds to Healthy (0), Mild (1), Moderate (2), Severe (3), and Rotten (4). Because misclassifying Grade 0 as Grade 4 carries far greater agricultural and economic penalty than misclassifying Grade 0 as Grade 1, multi-class categorical cross-entropy is inappropriate.

We employ Rank-Consistent Ordinal Regression (CORAL) \cite{cao2020rank}. The ordinal task is decomposed into $K-1$ binary classification sub-tasks sharing a single projection vector $\mathbf{w} \in \mathbb{R}^D$ and task-specific cutpoint thresholds $b_k$:

$$h_k(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b_k) \approx P(y > k \mid \mathbf{x}), \quad k \in \{0, \dots, K-2\}$$

## 2. Monotonic CORAL Parameterization

In conventional CORAL implementations, cutpoints $b_k$ are updated unconstrained via stochastic gradients. Across random seeds, cutpoint inversions ($b_{k} > b_{k-1}$) frequently occur, producing invalid cumulative distributions where $P(y > k) > P(y > k-1)$ and yielding negative class probabilities.

To guarantee mathematical consistency, we parameterize the cutpoint biases strictly via cumulative softplus decrements:

$$b_0 = \theta_0$$
$$b_k = \theta_0 - \sum_{j=1}^k \text{softplus}(\delta_j), \quad k \in \{1, \dots, K-2\}$$

Because $\text{softplus}(\delta) = \ln(1 + e^\delta) > 0$ strictly for all real $\delta \in \mathbb{R}$, this parameterization ensures:

$$b_0 > b_1 > b_2 > \dots > b_{K-2} \quad \text{unconditionally.}$$

The discrete class probabilities $P(y = k \mid \mathbf{x})$ are reconstructed via:

$$P(y = 0 \mid \mathbf{x}) = 1 - P(y > 0 \mid \mathbf{x})$$
$$P(y = k \mid \mathbf{x}) = P(y > k-1 \mid \mathbf{x}) - P(y > k \mid \mathbf{x}), \quad k \in \{1, \dots, K-2\}$$
$$P(y = K-1 \mid \mathbf{x}) = P(y > K-2 \mid \mathbf{x})$$

This formulation guarantees that $P(y = k \mid \mathbf{x}) \ge 0$ for all $k$ and $\sum_{k=0}^{K-1} P(y = k \mid \mathbf{x}) = 1.0$ within floating-point tolerance.

## 3. Class-Balanced Effective Number Weighting

To combat severe class imbalance ($36,414$ Healthy samples vs $208$ Moderate samples in training) without the extreme instability of raw inverse weighting, we adopt the Effective Number of Samples weighting formulation \cite{cui2019class}:

$$E_{n_k} = \frac{1 - \beta^{n_k}}{1 - \beta}, \quad w_k = \frac{1}{E_{n_k}}, \quad \text{with } \beta = 0.999$$

Weights are normalized such that $\sum_{k=0}^{K-1} w_k = K$. This bounds the maximum-to-minimum weight ratio to $5.32\times$, in stark contrast to the $175.07\times$ ratio of raw inverse frequency weighting.

## 4. Multi-Modal Context Conditioning Architectures

We formalize four comparative model architectures:
- **Configuration A (Image Only):** Frozen ConvNeXt-Tiny embeddings (768-D) $\to$ 2-layer MLP projection $\to$ MonotonicCORALHead.
- **Configuration B (Image + Species):** Visual projection (256-D) concatenated with 16-D species embedding from `RobustSpeciesEmbedding`. Unknown species map to index 5.
- **Configuration C (Image + OOD):** Visual projection (256-D) concatenated with 2-D Mahalanobis distance representation.
- **Configuration D (Image + Species + OOD):** Full multi-modal concatenation ($256 + 16 + 2 = 274\text{-D}$) $\to$ Fusion MLP $\to$ MonotonicCORALHead.

### Feature Extraction and Frozen Backbone Protocol

All 120 confirmatory experiments operate on precomputed ConvNeXt-Tiny feature tensors extracted once and cached in `outputs/features/phase4_dataset.pt` (56,036 samples × 768 dimensions, 179 MB). The ConvNeXt-Tiny backbone (~28.6M parameters) is not instantiated during training; no gradient updates flow through the visual encoder.

Only the Monotonic CORAL head and fusion MLP (~626K–630K parameters depending on configuration) receive gradient updates during training (AdamW, lr=1e-3, weight_decay=1e-4, 15 epochs, batch_size=2048 feature vectors).

This protocol was chosen to: (a) ensure fair comparison across configurations with identical visual representations, (b) reduce training cost for the 120-run confirmatory study on consumer GPU hardware (RTX 3050 6GB), and (c) isolate the effect of context conditioning from backbone initialization variance.

## 5. Dataset Partitions and Session Allocation

The dataset comprises 56,036 images partitioned into train (39,790), validation (3,413), and test (12,833) splits (`split_v2`). To resolve the complete absence of Grade 2 evaluation samples in historical splits, all 119 Banana Grade 2 images from acquisition session `20200730` are exclusively assigned to the test partition. Images from the same session belonging to other species-grade combinations (Apple Grade 0, $n=147$) remain in the training partition.

Full session-level holdout was not enforced because session identifiers in this dataset do not align with individual species or batches. Rather, within-species, within-grade session exclusivity is guaranteed for the Grade 2 Banana evaluation subset, ensuring zero image-level or identical fruit-grade leakage between partitions.
