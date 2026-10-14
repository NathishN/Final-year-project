# Task 2: Ordinal Performance Analysis

_Phase 3: CORAL Rank-Consistent Ordinal Regression_

---

## 2.1 Primary Evaluation Metrics

> Metrics not yet available.

## 2.2 Why QWK is the Primary Metric

**Quadratic Weighted Kappa** penalizes disagreements quadratically with
ordinal distance. For K=3 grades, the weight matrix is:

| True\Pred | Grade 1 | Grade 2 | Grade 3 |
|------------|---------|---------|---------|
| Grade 1    |  0.00   |  0.25   |  1.00   |
| Grade 2    |  0.25   |  0.00   |  0.25   |
| Grade 3    |  1.00   |  0.25   |  0.00   |

A Grade1->Grade3 error is penalized **4x more** than a Grade1->Grade2 error.
CORAL directly optimizes for this property via the shared-weight ordinal rank
structure, whereas standard CE treats all misclassifications equally.

**QWK Interpretation Scale:**

| QWK Range | Agreement Level     | Publication Threshold   |
|-----------|---------------------|-------------------------|
| < 0.2     | Slight              | Not publishable         |
| 0.2-0.4   | Fair                | Very weak               |
| 0.4-0.6   | Moderate            | Acceptable with caveats |
| 0.6-0.8   | Substantial         | Good                    |
| > 0.8     | Almost perfect      | Strong                  |

## 2.3 Why CORAL Succeeds for Agricultural Grading

CORAL enforces the rank-consistency constraint:

    P(y > 0 | x) >= P(y > 1 | x)

via a shared weight vector w. Predicted probability for rank k:

    P_hat(y > k | x) = sigmoid(w^T f(x) + b_k)

**Agricultural Significance:**
Grade boundaries are inherently cumulative:
- A Grade 3 fruit has ALL quality deficiencies of Grade 2 AND MORE.
- A Grade 1 fruit has NONE of the Grade 2 deficiencies.

This monotonic degradation maps exactly to the CORAL assumption, unlike
standard CE which treats grades as independent categories.

**Expected failure modes:**
1. **Grade 2 (Retail)** is hardest -- it is the boundary class, with features
   overlapping both Grade 1 and Grade 3.
2. **Grade 3 (Processing)** has the smallest sample count (355) and may have
   high false-negative rate if the model maps ambiguous fruits to Grade 2.
## 2.4 Classification Report

```
Not yet available.
```

## 2.5 Failure Mode Taxonomy

| Error Type          | Severity     | Cause                             | Mitigation                        |
|---------------------|--------------|-----------------------------------|-----------------------------------|
| Grade1 -> Grade2    | Low          | Boundary ambiguity                | Off-by-one acceptable             |
| Grade2 -> Grade1    | Low          | Optimistic bias                   | Check class weights               |
| Grade2 -> Grade3    | Moderate     | Minority overfit                  | More grade_3 samples              |
| Grade3 -> Grade2    | Moderate     | Grade 3 underrepresentation       | WeightedSampler                   |
| Grade1 -> Grade3    | **Critical** | Feature collapse                  | Inspect embeddings                |
| Grade3 -> Grade1    | **Critical** | Label noise / OOD                 | Label audit required              |

Critical errors (|error|>1) are catastrophic in commercial settings: a
Premium fruit misgraded as Processing results in direct economic loss.
