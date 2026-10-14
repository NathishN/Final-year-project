# Phase 5.2 — Methodological Decisions Log

### DEC-A1: Run Verification
- **Choice:** Accept 120/120 runs as VERIFIED_COMPLETE for inference/prediction artifacts
- **Rationale:** All 120 checkpoints and predictions exist, load, and reproduce QWK exactly.

### DEC-A2: Training Protocol Disclosure
- **Choice:** Explicitly mandate frozen-feature disclosure in manuscript
- **Rationale:** Backbone was frozen; head-only training must be stated.

### DEC-B1: Wilcoxon Statistic
- **Choice:** Adopt W=72, p=0.000555
- **Rationale:** Verified against SciPy 1.13 implementation on 30 paired seed differences.

### DEC-C1: Session Holdout Protocol
- **Choice:** Reassign session 20200730 entirely to test partition
- **Rationale:** Achieves 100% group isolation across all 80 sessions without synthesizing artificial data.

### DEC-D1: OOD Causal Claim
- **Choice:** Classify background causality as UNRESOLVED / CORRELATIONAL
- **Rationale:** Masked counterfactual ablation cannot be proven without ground-truth segmentation masks.

