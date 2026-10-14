# Task 4: Failure Case Analysis

---

> Predictions not yet available. Run training first.

## 4.5 Root Cause Analysis

**Grade 1 -> Grade 3 errors:**
Root cause: Likely label noise OR session-specific lighting/background shifts
that perturb CORAL threshold values.
Mitigation: Semi-supervised label refinement; remove predictions with
CORAL confidence < 0.6 from training.

**Grade 3 -> Grade 1 errors:**
Root cause: Grade 3 fruits that appear visually similar to Grade 1 but have
internal defects not captured by RGB imaging. This is a fundamental limit of
RGB-only input.
Mitigation: Multi-modal input (NIR, depth) or ensemble with texture
analysis network.

**High confidence failures:**
When sigmoid(g(x)+b_k) > 0.9 but the grade is wrong, the model has
memorized spurious cues (e.g., background color, image artifacts).
Mitigation: GradCAM visualization to confirm the model attends to actual
fruit surface, not background.
