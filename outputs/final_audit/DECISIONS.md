# Master Audit — Scientific & Methodological Decisions

This document logs all key scientific, architectural, and statistical decisions made during the final project audit and completion.

## Decision 1: Protocol Standardization (split_v2)
- **Decision:** Use `split_v2` with session `20200730` reallocated to the test set as the primary 5-grade evaluation benchmark.
- **Rationale:** Historical `split_v1` had zero Grade 2 test examples ($n=0$), rendering 5-grade evaluation mathematically impossible. `split_v2` introduces 119 genuine Grade 2 test images from independent specimens with verified zero cross-split leakage.

## Decision 2: Monotonic Threshold Constraint
- **Decision:** Implement `MonotonicCORALHead` with strictly decreasing cutpoint biases ($b_0 > b_1 > \dots > b_{K-2}$) parameterized via cumulative softplus decrements.
- **Rationale:** Historical unconstrained CORAL thresholds inverted across random seeds, producing negative class probabilities and undefined cumulative CDFs.

## Decision 3: Loss Weighting Boundedness
- **Decision:** Adopt Effective Number of Samples weighting (Cui et al., CVPR 2019, $\beta=0.999$) calculated strictly on the training partition.
- **Rationale:** Raw inverse frequency weighting produced an extreme 175:1 weight ratio, causing false-alarm explosions on Grade 2. Effective Number weighting bounds the ratio to 5.32:1.

## Decision 4: Real-Image OOD Evaluation
- **Decision:** Reject synthetic Gaussian AUROC claims. Extract 5,000 real images across 5 unseen species (*Cherry, Kiwi, Peach, Pineapple, Plum*) using the official ConvNeXt-Tiny backbone.
- **Rationale:** The historical 100% AUROC claim was an artifact of synthetic Gaussian feature distributions.

## Decision 5: Non-Cherry-Picked Confirmatory Reporting
- **Decision:** Report all 30 prespecified seeds deterministically across all 4 configurations ($N=120$ total runs). Transparently report that species conditioning degrades QWK ($\Delta\text{QWK} = -0.0360, p < 0.001$).
- **Rationale:** Scientific integrity requires accepting and publishing valid negative findings rather than forcing an unsupported hypothesis.
