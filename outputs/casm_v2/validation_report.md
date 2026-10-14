# CASM v2 Validation & Correctness Test Report

| Test Case | Status | Detail / Output |
|---|---|---|
| Forward Pass & Shape | PASS | Logits (8,4), Ranks (8,) correctly shaped |
| Unknown Species Fallback | PASS | Out-of-bounds species IDs safely mapped to fallback embedding |
| CORAL Encoding & Loss | PASS | CORALLoss returns finite scalar: 0.7400 |
| Ordinal Probability Bounds | PASS | All cumulative probabilities bounded in [0, 1] |
| Gradient Propagation | PASS | Loss backward pass succeeds with finite gradient norm |
| CPU/CUDA Numerical Parity | PASS | Max logit discrepancy = 2.38e-07 |
| Checkpoint Save & Load | PASS | Model state_dict saved and restored cleanly |
| Seed Reproducibility | PASS | Identical seed produces identical weight initialization |
| Split Isolation Audit | PASS | 0 overlapping image paths across train (39909), val (3413), test (12714) |
