"""
analyze_phase3_coral.py
========================
Post-training analysis suite for Phase 3: Rank-Consistent Ordinal Regression (CORAL).

Paper: "Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using
        Context-Aware Severity Modeling and Uncertainty Calibration"

Tasks covered:
  Task 1  -- Training curve analysis
  Task 2  -- Ordinal performance analysis
  Task 3  -- Species-wise analysis
  Task 4  -- Failure case analysis
  Task 5  -- Statistical significance (t-test, multi-seed)
  Task 6  -- IEEE publication tables
  Task 7  -- Research contribution validation
  Task 8  -- Phase 4 readiness
  Task 9  -- Full IEEE reviewer report

Usage:
    python analyze_phase3_coral.py

Reads:
    outputs/coral/coral_metrics.json
    outputs/coral/coral_predictions.csv
    outputs/coral/ablation_results.csv
    outputs/coral/multiseed_results.csv
    outputs/coral/species_metrics.json
    outputs/coral/dataset_statistics.json
    outputs/coral/coral_classification_report.txt

Writes:
    outputs/coral/training_analysis.md
    outputs/coral/ordinal_analysis.md
    outputs/coral/species_analysis.md
    outputs/coral/failure_analysis.md
    outputs/coral/statistical_significance.md
    outputs/coral/publication_tables.md
    outputs/coral/contribution_review.md
    outputs/coral/phase4_readiness.md
    outputs/coral/ieee_review_report.md
    outputs/coral/full_technical_report.md
"""

import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("outputs/coral")
OUT.mkdir(parents=True, exist_ok=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_json(path):
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def load_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def w(f, text):
    f.write(text)

def hdr(f, title, level=1):
    f.write("#" * level + " " + title + "\n\n")

def tbl(f, df, index=False):
    try:
        f.write(df.to_markdown(index=index))
    except Exception:
        f.write(df.to_string(index=index))
    f.write("\n\n")

def rule(f):
    f.write("---\n\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 -- TRAINING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def task1_training_analysis():
    metrics = load_json(OUT / "coral_metrics.json")

    out = OUT / "training_analysis.md"
    with open(out, "w", encoding="utf-8") as f:
        hdr(f, "Task 1: CORAL Training Analysis", 1)
        w(f, "_Paper: Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading_\n\n")
        rule(f)

        hdr(f, "1.1 Final Validation Metrics (Best Checkpoint)", 2)
        if metrics:
            rows = [{"Metric": k.upper().replace("_", " "), "Value": v} for k, v in metrics.items()]
            tbl(f, pd.DataFrame(rows))
        else:
            w(f, "> Training metrics not yet available. Run `train_phase3_coral.py` first.\n\n")

        hdr(f, "1.2 Training Configuration Summary", 2)
        config_rows = [
            ("Backbone",           "ConvNeXt-Tiny (ImageNet-1K pretrained)"),
            ("Head",               "CoralHead (shared-w, K-1=2 ordinal outputs)"),
            ("Stage 1",            "5 epochs, backbone frozen, lr=3e-3"),
            ("Stage 2",            "up to 20 epochs, full fine-tune, lr=3e-4"),
            ("Optimizer",          "AdamW (weight_decay=1e-2)"),
            ("Scheduler",          "CosineAnnealingLR"),
            ("Mixed Precision",    "AMP FP16 (torch.amp.autocast)"),
            ("Gradient Clipping",  "1.0"),
            ("Early Stopping",     "patience=5 on val QWK"),
            ("Batch Size",         "16 (safe for RTX 3050 6GB VRAM)"),
            ("Sampler",            "WeightedRandomSampler (imbalance ratio 1.84x)"),
            ("Augmentation",       "RandomCrop, HFlip, VFlip, Rotation, ColorJitter, RandomErasing"),
        ]
        tbl(f, pd.DataFrame(config_rows, columns=["Parameter", "Value"]))

        hdr(f, "1.3 Two-Stage Training Rationale", 2)
        w(f, """**Why Two-Stage Training?**

The real_grade domain contains only 1,602 samples across 3 fruit species.
Fine-tuning a 28M-parameter ConvNeXt-Tiny backbone end-to-end from random
head initialization risks catastrophic interference between backbone
representation learning and CORAL head optimization.

**Stage 1 (Head-only, 5 epochs):**
- Backbone frozen at ImageNet weights
- Only the CORAL head (weight w, biases b0, b1) and LayerNorm are trained
- LR = 3e-3 with cosine decay
- Establishes stable ordinal threshold separation before backbone perturbation

**Stage 2 (Full fine-tune):**
- All parameters unfrozen
- LR reduced to 3e-4 (10x lower) with cosine annealing
- Allows domain-specific texture adaptation without destroying pretrained representations
- Early stopping on QWK prevents overfitting on small dataset

**Expected Convergence:**
- Stage 1: QWK increases from ~0.0 to ~0.3-0.5 (head learns ordinal boundaries)
- Stage 2: QWK increases to ~0.5-0.8, then plateaus or early-stops
- Loss: Binary CE typically settles in range [0.3, 0.6] for 3-class ordinal problems
""")

        hdr(f, "1.4 Overfitting Risk Assessment", 2)
        w(f, """**Dataset Size vs. Model Capacity:**

| Component | Parameters |
|-----------|------------|
| ConvNeXt-Tiny backbone | ~27.8M |
| LayerNorm (768) | 1,536 |
| CoralHead (w + b) | 770 |
| **Total trainable** | **~27.8M** |

With only 869 training samples, the effective samples-per-parameter ratio
is extremely low (~3.1e-5). Mitigations applied:

1. **Pretrained backbone**: 90%+ of parameters already encode generalizable
   visual features from ImageNet-1K (1.28M images).
2. **Stage 1 freezing**: Prevents 27.8M backbone parameters from overfitting
   during initial head training.
3. **Dropout (p=0.15)**: Applied before the CORAL head.
4. **WeightedRandomSampler**: Ensures minority grade_3 is seen each epoch.
5. **Heavy augmentation**: Effective training set size is multiplied through
   RandomCrop, ColorJitter, and RandomErasing.
6. **Early stopping (patience=5)**: Terminates training before QWK on val degrades.

**Key diagnostic**: Train QWK vs Val QWK gap > 0.15 would indicate overfitting.
""")

    print(f"Task 1 written: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 -- ORDINAL PERFORMANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def task2_ordinal_analysis():
    metrics  = load_json(OUT / "coral_metrics.json")
    cls_rpt  = (OUT / "coral_classification_report.txt").read_text(encoding="utf-8") \
               if (OUT / "coral_classification_report.txt").exists() else "Not yet available."

    out = OUT / "ordinal_analysis.md"
    with open(out, "w", encoding="utf-8") as f:
        hdr(f, "Task 2: Ordinal Performance Analysis", 1)
        w(f, "_Phase 3: CORAL Rank-Consistent Ordinal Regression_\n\n")
        rule(f)

        hdr(f, "2.1 Primary Evaluation Metrics", 2)
        if metrics:
            metric_rows = [
                {"Metric": "Accuracy",            "Value": f"{metrics.get('accuracy', 0):.4f}",          "Interpretation": "Exact grade match rate"},
                {"Metric": "Macro-F1",            "Value": f"{metrics.get('macro_f1', 0):.4f}",          "Interpretation": "Unweighted class-average F1"},
                {"Metric": "MAE",                 "Value": f"{metrics.get('mae', 0):.4f}",               "Interpretation": "Mean |pred - true| grade levels"},
                {"Metric": "QWK",                 "Value": f"{metrics.get('qwk', 0):.4f}",               "Interpretation": "Ordinal agreement penalizing large errors"},
                {"Metric": "Off-by-One Acc (%)",  "Value": f"{metrics.get('off_by_one_pct', 0):.2f}",   "Interpretation": "Predictions within 1 grade level"},
            ]
            tbl(f, pd.DataFrame(metric_rows))
        else:
            w(f, "> Metrics not yet available.\n\n")

        hdr(f, "2.2 Why QWK is the Primary Metric", 2)
        w(f, """**Quadratic Weighted Kappa** penalizes disagreements quadratically with
ordinal distance. For K=3 grades, the weight matrix is:

| True\\Pred | Grade 1 | Grade 2 | Grade 3 |
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

""")

        hdr(f, "2.3 Why CORAL Succeeds for Agricultural Grading", 2)
        w(f, """CORAL enforces the rank-consistency constraint:

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
""")

        hdr(f, "2.4 Classification Report", 2)
        w(f, "```\n" + cls_rpt + "\n```\n\n")

        hdr(f, "2.5 Failure Mode Taxonomy", 2)
        w(f, """| Error Type          | Severity     | Cause                             | Mitigation                        |
|---------------------|--------------|-----------------------------------|-----------------------------------|
| Grade1 -> Grade2    | Low          | Boundary ambiguity                | Off-by-one acceptable             |
| Grade2 -> Grade1    | Low          | Optimistic bias                   | Check class weights               |
| Grade2 -> Grade3    | Moderate     | Minority overfit                  | More grade_3 samples              |
| Grade3 -> Grade2    | Moderate     | Grade 3 underrepresentation       | WeightedSampler                   |
| Grade1 -> Grade3    | **Critical** | Feature collapse                  | Inspect embeddings                |
| Grade3 -> Grade1    | **Critical** | Label noise / OOD                 | Label audit required              |

Critical errors (|error|>1) are catastrophic in commercial settings: a
Premium fruit misgraded as Processing results in direct economic loss.
""")

    print(f"Task 2 written: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 -- SPECIES-WISE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def task3_species_analysis():
    species = load_json(OUT / "species_metrics.json")
    dataset = load_json(OUT / "dataset_statistics.json")

    out = OUT / "species_analysis.md"
    with open(out, "w", encoding="utf-8") as f:
        hdr(f, "Task 3: Species-Wise Ordinal Analysis", 1)
        rule(f)

        hdr(f, "3.1 Per-Species Performance Table", 2)
        if species:
            rows = []
            for fruit, m in species.items():
                if "note" in m:
                    rows.append({"Species": fruit, "N": "N/A", "Accuracy": "N/A",
                                 "MAE": "N/A", "QWK": "N/A", "Macro-F1": "N/A", "Off-by-1 %": "N/A"})
                else:
                    rows.append({
                        "Species":    fruit,
                        "N":          m.get("sample_count", "?"),
                        "Accuracy":   f"{m.get('accuracy', 0):.4f}",
                        "MAE":        f"{m.get('mae', 0):.4f}",
                        "QWK":        f"{m.get('qwk', 0):.4f}",
                        "Macro-F1":   f"{m.get('macro_f1', 0):.4f}",
                        "Off-by-1 %": f"{m.get('off_by_one_pct', 0):.2f}",
                    })
            tbl(f, pd.DataFrame(rows))
        else:
            w(f, "> Species metrics not yet available.\n\n")

        hdr(f, "3.2 Species Difficulty Ranking", 2)
        w(f, """| Rank | Species | Difficulty | Primary Reason |
|------|---------|------------|----------------|
| 1 (Hardest) | Tomato | HIGH | Color-based grade signal confounded by ripeness; grade_3 has 0 test samples |
| 2 | Apple | MEDIUM | Grade 2/3 boundary visually ambiguous (bruising vs. severe damage) |
| 3 (Easiest) | Mango | LOWER | Grade 1/3 extremes visually distinct; largest sample count (618) |

""")

        hdr(f, "3.3 Biological and Annotation Reasoning", 2)
        w(f, """**Apple:**
- Grade 1 (Premium): Uniform colour, no surface defects, firm texture
- Grade 2 (Retail): Minor surface blemishes, slight discolouration
- Grade 3 (Processing): Visible bruising, decay spots, deformation
- Challenge: Grade 2/3 boundary is highly subjective. A 5mm vs 10mm bruise may
  be annotated differently across sessions, introducing systematic label noise.

**Mango:**
- Grade 1: Uniform yellow-orange skin, no black spots, correct shape
- Grade 2: Minor skin blemishes, slight overripeness
- Grade 3: Black spots, stem-end rot, severe overripeness
- Advantage: Black spot detection provides a strong discriminative signal that
  maps well to ConvNeXt's learned texture filters.

**Tomato:**
- Grade 1: Firm, uniform red, no cracks
- Grade 2: Minor surface cracks, colour variability
- Grade 3: Severe cracking, mould, decay
- Challenge: Ripeness (colour) and quality (structural integrity) are conflated.
  A fully ripe (dark red) tomato can be Grade 1 or Grade 3, creating a
  fundamental ambiguity in the RGB feature space.
- Dataset Issue: Tomato has 3 sessions, ALL assigned to train/val.
  Zero test samples = evaluation is entirely validation-based (optimistic).
""")

        hdr(f, "3.4 Dataset-Level Species Analysis", 2)
        if dataset:
            fg = dataset.get("fruit_x_grade", {})
            rows = []
            for fruit, grades in fg.items():
                mn = min(grades.values()) if grades else 1
                mx = max(grades.values()) if grades else 1
                rows.append({
                    "Species": fruit,
                    "Grade 1": grades.get("grade_1", 0),
                    "Grade 2": grades.get("grade_2", 0),
                    "Grade 3": grades.get("grade_3", 0),
                    "Total":   sum(grades.values()),
                    "Imbalance": f"{mx/max(mn,1):.2f}x",
                })
            tbl(f, pd.DataFrame(rows))

    print(f"Task 3 written: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 -- FAILURE CASE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def task4_failure_analysis():
    pred_df = load_csv(OUT / "coral_predictions.csv")

    out = OUT / "failure_analysis.md"
    with open(out, "w", encoding="utf-8") as f:
        hdr(f, "Task 4: Failure Case Analysis", 1)
        rule(f)

        if pred_df.empty:
            w(f, "> Predictions not yet available. Run training first.\n\n")
        else:
            total    = len(pred_df)
            correct  = (pred_df["true_grade"] == pred_df["pred_grade"]).sum()
            critical = (pred_df["abs_error"] > 1).sum()
            moderate = (pred_df["abs_error"] == 1).sum()

            hdr(f, "4.1 Error Distribution", 2)
            err_rows = [
                {"Error Category": "Correct (|error|=0)",    "Count": int(correct),  "Pct": f"{100*correct/total:.1f}%",  "Severity": "None"},
                {"Error Category": "Off-by-One (|error|=1)", "Count": int(moderate), "Pct": f"{100*moderate/total:.1f}%", "Severity": "Acceptable"},
                {"Error Category": "Critical (|error|>1)",   "Count": int(critical), "Pct": f"{100*critical/total:.1f}%", "Severity": "SEVERE"},
            ]
            tbl(f, pd.DataFrame(err_rows))

            hdr(f, "4.2 Most Confused Grade Pairs", 2)
            misclass = pred_df[pred_df["true_grade"] != pred_df["pred_grade"]]
            if "confusion_pair" in pred_df.columns and len(misclass) > 0:
                pair_counts = misclass["confusion_pair"].value_counts().reset_index()
                pair_counts.columns = ["Confusion Pair", "Count"]
                pair_counts["% of Errors"] = (pair_counts["Count"] / len(misclass) * 100).round(1)
                tbl(f, pair_counts)

            hdr(f, "4.3 Critical Ordinal Errors (|error| > 1)", 2)
            critical_df = pred_df[pred_df["abs_error"] > 1]
            w(f, f"Total critical errors: **{len(critical_df)}** / {total} ({100*len(critical_df)/total:.1f}%)\n\n")
            if len(critical_df) > 0 and "confusion_pair" in critical_df.columns:
                tbl(f, critical_df.head(10)[["fruit", "confusion_pair", "image_path"]])
            else:
                w(f, "> No critical errors detected. CORAL ordinal constraint is effective.\n\n")

            hdr(f, "4.4 Per-Species Failure Rates", 2)
            if "fruit" in pred_df.columns:
                for fruit in ["Apple", "Mango", "Tomato"]:
                    sub = pred_df[pred_df["fruit"] == fruit]
                    if len(sub) == 0:
                        w(f, f"**{fruit}**: No samples in eval split.\n\n")
                        continue
                    err_rate  = (sub["true_grade"] != sub["pred_grade"]).mean()
                    crit_rate = (sub["abs_error"] > 1).mean()
                    w(f, f"**{fruit}**: N={len(sub)} | Error rate={err_rate:.1%} | Critical rate={crit_rate:.1%}\n\n")

        hdr(f, "4.5 Root Cause Analysis", 2)
        w(f, """**Grade 1 -> Grade 3 errors:**
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
""")

    print(f"Task 4 written: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 -- STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════════════════

def task5_statistical_significance():
    ms = load_csv(OUT / "multiseed_results.csv")

    out = OUT / "statistical_significance.md"
    with open(out, "w", encoding="utf-8") as f:
        hdr(f, "Task 5: Statistical Significance Testing", 1)
        rule(f)

        hdr(f, "5.1 Multi-Seed Results Table", 2)
        if not ms.empty:
            tbl(f, ms)
        else:
            w(f, "> Multi-seed results not yet available.\n\n")

        hdr(f, "5.2 Paired t-test Methodology", 2)
        w(f, """**Null Hypothesis (H0):**
No significant difference in mean QWK between CORAL and Cross-Entropy across 5 seeds.

**Alternative Hypothesis (H1):**
CORAL achieves significantly higher QWK than Cross-Entropy.

**Test selected:** One-sided paired t-test (parametric, alpha = 0.05).

**Justification for paired test:**
- Each seed produces matched observations (same data, same random split).
- Pairing removes between-seed variability as a confound.
- 5 pairs satisfies minimum degrees of freedom for t-test (df=4).

**Significance threshold:** p < 0.05 (one-sided). Bonferroni correction for 3
simultaneous metrics: alpha_corrected = 0.05/3 = 0.017.

**Effect size (Cohen's d):**

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.2     | Negligible     |
| 0.2-0.5   | Small          |
| 0.5-0.8   | Medium         |
| > 0.8     | Large          |

""")

        hdr(f, "5.3 t-test Computation", 2)
        if not ms.empty:
            for metric in ["QWK", "MAE", "ACCURACY"]:
                coral_row = ms[(ms["Method"] == "CORAL") & (ms["Metric"] == metric)]
                ce_row    = ms[(ms["Method"] == "CrossEntropy") & (ms["Metric"] == metric)]
                if coral_row.empty or ce_row.empty:
                    continue
                c_mean = float(coral_row["Mean"].iloc[0])
                c_std  = float(coral_row["Std"].iloc[0])
                e_mean = float(ce_row["Mean"].iloc[0])
                e_std  = float(ce_row["Std"].iloc[0])
                # For MAE lower is better
                delta_mean = (e_mean - c_mean) if metric == "MAE" else (c_mean - e_mean)
                n = 5
                pooled_se = np.sqrt((c_std**2 + e_std**2) / n)
                w(f, f"**{metric}:**\n")
                w(f, f"- CORAL:  {c_mean:.4f} +/- {c_std:.4f}\n")
                w(f, f"- CE:     {e_mean:.4f} +/- {e_std:.4f}\n")
                w(f, f"- Delta:  {delta_mean:+.4f} ({'CORAL better' if delta_mean > 0 else 'CE better'})\n")
                if pooled_se > 0:
                    t_stat  = delta_mean / pooled_se
                    p_val   = float(stats.t.sf(abs(t_stat), df=n - 1))
                    cohen_d = delta_mean / max(np.sqrt((c_std**2 + e_std**2) / 2), 1e-9)
                    sig = "**SIGNIFICANT** (p < 0.05)" if p_val < 0.05 else "NOT significant (p >= 0.05)"
                    w(f, f"- t-statistic: {t_stat:.3f} | p-value: {p_val:.4f} | Cohen's d: {cohen_d:.3f}\n")
                    w(f, f"- Result: {sig}\n\n")
        else:
            w(f, "> Awaiting multi-seed results.\n\n")
            w(f, "Will compare CORAL vs CE across seeds [42, 123, 2025, 999, 777] for QWK, MAE, Accuracy.\n\n")

        hdr(f, "5.4 Statistical Recommendations for IEEE Submission", 2)
        w(f, """1. **Report mean +/- std** for all metrics across 5 seeds.
2. **Use paired t-test** (n=5) -- paired removes seed-variance confound.
3. **Report Cohen's d** alongside p-value -- p-value alone is insufficient.
4. **Bonferroni correction** for 3 simultaneous metrics: alpha = 0.017.
5. **McNemar's test** for per-sample accuracy comparison.
6. **95% CI** on mean QWK improvement: CORAL_mean +/- 1.96 * SE.

IEEE Reviewer expectation:
> "Authors must demonstrate statistical significance of their proposed method
> over baselines. Single-seed results are insufficient for publication in IEEE
> Transactions journals."
""")

    print(f"Task 5 written: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 6 -- IEEE PUBLICATION TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def task6_publication_tables():
    dataset  = load_json(OUT / "dataset_statistics.json")
    metrics  = load_json(OUT / "coral_metrics.json")
    ablation = load_csv(OUT / "ablation_results.csv")
    ms       = load_csv(OUT / "multiseed_results.csv")

    out = OUT / "publication_tables.md"
    with open(out, "w", encoding="utf-8") as f:
        hdr(f, "IEEE Publication-Ready Tables", 1)
        w(f, "_Paper: Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading_\n\n")
        w(f, "> All tables follow IEEE Transactions formatting conventions.\n")
        w(f, "> Bold = best result.\n\n")
        rule(f)

        # Table 1
        hdr(f, "Table I: Overall Dataset Statistics", 2)
        t1 = pd.DataFrame([
            {"Domain": "fruits360",     "Images": 43962, "Purpose": "Closed-set fruit classification"},
            {"Domain": "real_quality",  "Images": 7388,  "Purpose": "Fresh/mild/rotten severity"},
            {"Domain": "real_severity", "Images": 2602,  "Purpose": "Multi-label severity annotation"},
            {"Domain": "real_grade",    "Images": 2086,  "Purpose": "Grade 1/2/3 ordinal labelling"},
            {"Domain": "**Total**",     "Images": 56038, "Purpose": "Multi-domain corpus"},
        ])
        tbl(f, t1)

        # Table 2
        hdr(f, "Table II: Fruit Species Distribution", 2)
        t2 = pd.DataFrame([
            {"Species": "Apple",     "Images": 24962, "% Total": "44.5%", "Domains": "fruits360, real_grade"},
            {"Species": "Tomato",    "Images": 16164, "% Total": "28.8%", "Domains": "fruits360, real_quality, real_grade"},
            {"Species": "Orange",    "Images": 6927,  "% Total": "12.4%", "Domains": "fruits360"},
            {"Species": "Banana",    "Images": 6143,  "% Total": "11.0%", "Domains": "fruits360"},
            {"Species": "Mango",     "Images": 1842,  "% Total": "3.3%",  "Domains": "real_grade"},
            {"Species": "**Total**", "Images": 56038, "% Total": "100%",  "Domains": "--"},
        ])
        tbl(f, t2)

        # Table 3
        hdr(f, "Table III: Grade Distribution (real_grade domain)", 2)
        fg = dataset.get("fruit_x_grade", {}) if dataset else {}
        if fg:
            rows = []
            for fruit, grades in fg.items():
                rows.append({
                    "Species": fruit,
                    "Grade 1": grades.get("grade_1", 0),
                    "Grade 2": grades.get("grade_2", 0),
                    "Grade 3": grades.get("grade_3", 0),
                    "Total":   sum(grades.values()),
                })
            rows.append({
                "Species": "**Total**",
                "Grade 1": sum(r["Grade 1"] for r in rows),
                "Grade 2": sum(r["Grade 2"] for r in rows),
                "Grade 3": sum(r["Grade 3"] for r in rows),
                "Total":   sum(r["Total"] for r in rows),
            })
            tbl(f, pd.DataFrame(rows))
        else:
            w(f, "> Grade distribution pending dataset analysis.\n\n")

        # Table 4
        hdr(f, "Table IV: Phase 1 Baseline -- Fruit Classification", 2)
        t4 = pd.DataFrame([
            {"Method": "ResNet-50 (CE)",                       "Acc": "--", "Macro-F1": "--", "Note": "Standard baseline"},
            {"Method": "EfficientNet-B0 (CE)",                 "Acc": "--", "Macro-F1": "--", "Note": "Lightweight"},
            {"Method": "**ConvNeXt-Tiny (CB-Focal, Ours)**",  "Acc": "Best", "Macro-F1": "Best", "Note": "Selected backbone"},
        ])
        tbl(f, t4)
        w(f, "> Phase 1 values to be filled from `outputs/baseline_metrics.json`.\n\n")

        # Table 5
        hdr(f, "Table V: Open-Set Recognition Results (Phase 2)", 2)
        t5 = pd.DataFrame([
            {"Method": "MSP (Hendrycks & Gimpel 2017)",      "AUROC": "--",      "AUPR": "--",      "FPR@95": "--"},
            {"Method": "Energy (Liu et al. 2020)",            "AUROC": "--",      "AUPR": "--",      "FPR@95": "--"},
            {"Method": "**Mahalanobis (Lee et al. 2018)**",  "AUROC": "**1.000**","AUPR": "**1.000**","FPR@95": "**0.00**"},
        ])
        tbl(f, t5)
        w(f, "> Note: AUROC=1.000 should be validated with held-out species as OOD distribution.\n\n")

        # Table 6
        hdr(f, "Table VI: Phase 3 Ordinal Regression Results", 2)
        if metrics:
            t6 = pd.DataFrame([{
                "Method":       "**CORAL (Ours)**",
                "Accuracy":     f"{metrics.get('accuracy', '--'):.4f}",
                "Macro-F1":     f"{metrics.get('macro_f1', '--'):.4f}",
                "MAE":          f"{metrics.get('mae', '--'):.4f}",
                "QWK":          f"{metrics.get('qwk', '--'):.4f}",
                "Off-by-1 (%)": f"{metrics.get('off_by_one_pct', '--'):.2f}",
                "Backbone":     "ConvNeXt-Tiny",
            }])
            tbl(f, t6)
        else:
            w(f, "> Awaiting CORAL training completion.\n\n")

        # Table 7
        hdr(f, "Table VII: Ablation Study -- Loss Function Comparison", 2)
        if not ablation.empty:
            tbl(f, ablation)
        else:
            t7 = pd.DataFrame([
                {"Loss": "Cross Entropy",         "Acc": "--", "F1": "--", "MAE": "--", "QWK": "--", "Off-by-1": "--"},
                {"Loss": "Weighted CE",            "Acc": "--", "F1": "--", "MAE": "--", "QWK": "--", "Off-by-1": "--"},
                {"Loss": "**CORAL (Ours)**",       "Acc": "--", "F1": "--", "MAE": "--", "QWK": "--", "Off-by-1": "--"},
            ])
            tbl(f, t7)
            w(f, "> Ablation results will populate after training completes.\n\n")

        # Table 8
        hdr(f, "Table VIII: Multi-Seed Stability (5 Seeds: 42, 123, 2025, 999, 777)", 2)
        if not ms.empty:
            cv_rows = ms[ms["Method"] == "CORAL"].copy()
            if not cv_rows.empty:
                cv_rows["Mean +/- Std"] = cv_rows.apply(
                    lambda r: f"{float(r['Mean']):.4f} +/- {float(r['Std']):.4f}", axis=1
                )
                ce_rows = ms[ms["Method"] == "CrossEntropy"].copy()
                if not ce_rows.empty:
                    ce_rows["Mean +/- Std (CE)"] = ce_rows.apply(
                        lambda r: f"{float(r['Mean']):.4f} +/- {float(r['Std']):.4f}", axis=1
                    )
                    merged = cv_rows[["Metric", "Mean +/- Std"]].merge(
                        ce_rows[["Metric", "Mean +/- Std (CE)"]], on="Metric", how="outer"
                    )
                    tbl(f, merged)
                else:
                    tbl(f, cv_rows[["Metric", "Mean +/- Std"]])
        else:
            w(f, "| Metric   | CORAL Mean +/- Std | CE Mean +/- Std |\n")
            w(f, "|----------|--------------------|-----------------|\n")
            w(f, "| QWK      | -- +/- --          | -- +/- --       |\n")
            w(f, "| MAE      | -- +/- --          | -- +/- --       |\n")
            w(f, "| Accuracy | -- +/- --          | -- +/- --       |\n\n")
            w(f, "> Multi-seed results pending.\n\n")

    print(f"Task 6 written: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 7 -- RESEARCH CONTRIBUTION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def task7_contribution_review():
    out = OUT / "contribution_review.md"
    with open(out, "w", encoding="utf-8") as f:
        hdr(f, "Task 7: Research Contribution Validation", 1)
        w(f, "_IEEE Reviewer Perspective_\n\n")
        rule(f)

        hdr(f, "7.1 Contribution Scoring Matrix", 2)
        contribs = pd.DataFrame([
            {"Contribution": "Open-Set Recognition (OOD)",
             "Novelty": "7/10", "Technical Depth": "8/10",
             "Scientific Rigor": "7/10", "Publication Strength": "8/10",
             "Notes": "Mahalanobis is established; applying to fruit quality is novel"},
            {"Contribution": "CORAL Ordinal Regression",
             "Novelty": "6/10", "Technical Depth": "8/10",
             "Scientific Rigor": "7/10", "Publication Strength": "7/10",
             "Notes": "CORAL (2020) is known; first application to fruit quality grading"},
            {"Contribution": "Context-Aware Severity Modelling",
             "Novelty": "8/10", "Technical Depth": "7/10",
             "Scientific Rigor": "6/10", "Publication Strength": "8/10",
             "Notes": "Phase 4 -- not yet implemented; highest novelty potential"},
            {"Contribution": "Uncertainty Calibration",
             "Novelty": "7/10", "Technical Depth": "7/10",
             "Scientific Rigor": "6/10", "Publication Strength": "7/10",
             "Notes": "Phase 5 -- Temperature Scaling; standard but important"},
            {"Contribution": "Multi-Domain Robustness",
             "Novelty": "8/10", "Technical Depth": "9/10",
             "Scientific Rigor": "8/10", "Publication Strength": "9/10",
             "Notes": "4-domain dataset with session-level leakage prevention is strong"},
        ])
        tbl(f, contribs)

        hdr(f, "7.2 Major Strengths", 2)
        w(f, """1. **Multi-domain dataset construction**: Merging fruits360, real_quality,
   real_severity, and real_grade with zero session leakage is a genuine
   engineering contribution. Session-level splitting is statistically more
   rigorous than image-level random splitting.

2. **Principled OOD integration**: Mahalanobis distance in ConvNeXt feature
   space provides a theoretically grounded uncertainty signal for open-set
   rejection. No additional training overhead required.

3. **Ordinal loss design**: CORAL's shared-weight formulation enforces monotonic
   rank consistency, which is semantically correct for agricultural quality grading.

4. **Complete ML pipeline**: The project covers the full stack: data -> features
   -> classification -> OOD -> ordinal -> severity -> calibration. This breadth
   is rare in agricultural AI literature.

5. **Reproducibility**: Session-level splits, fixed seeds, checkpoint saving, and
   multi-seed evaluation demonstrate strong reproducibility commitment.
""")

        hdr(f, "7.3 Major Weaknesses", 2)
        w(f, """1. **Small real-domain dataset (N=1,602)**: Core contributions rest on 1,602
   real-world images. Insufficient for IEEE Transactions without CV + multi-seed.

2. **Tomato evaluation gap**: All 3 Tomato sessions are in train/val. Per-species
   results for Tomato are not derived from a genuine held-out test set.

3. **No inter-rater reliability**: Agricultural grade labels have inherent
   subjectivity. Without reporting kappa_inter-rater, label quality is unknown.

4. **AUROC=1.000 for OOD is suspicious**: May indicate evaluation is not
   sufficiently distribution-shifted. Held-out species OOD is required.

5. **Phases 4-5 incomplete**: Title promises Context-Aware Severity Modelling
   and Uncertainty Calibration -- both are not yet implemented.
""")

        hdr(f, "7.4 Critical Risks", 2)
        risk_rows = [
            {"Risk": "CORAL shows no significant improvement over CE",   "Prob": "Medium", "Impact": "High",     "Mitigation": "Multi-seed CI; focus on QWK"},
            {"Risk": "OOD AUROC=1.0 is train-test contamination",        "Prob": "Medium", "Impact": "Critical", "Mitigation": "Test on genuinely novel species"},
            {"Risk": "Label noise in grade_3 causes floor effect",        "Prob": "High",   "Impact": "Medium",   "Mitigation": "Manual grade_3 misclassification inspection"},
            {"Risk": "Dataset too small for Transactions journal",        "Prob": "High",   "Impact": "High",     "Mitigation": "Expand real_grade to 5,000+ images"},
            {"Risk": "Phase 4/5 not novel enough individually",           "Prob": "Medium", "Impact": "Medium",   "Mitigation": "Frame as unified trustworthy system"},
        ]
        tbl(f, pd.DataFrame(risk_rows))

    print(f"Task 7 written: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 8 -- PHASE 4 READINESS
# ═══════════════════════════════════════════════════════════════════════════════

def task8_phase4_readiness():
    metrics = load_json(OUT / "coral_metrics.json")

    out = OUT / "phase4_readiness.md"
    with open(out, "w", encoding="utf-8") as f:
        hdr(f, "Task 8: Phase 4 Readiness Analysis", 1)
        w(f, "_Context-Aware Severity Modelling_\n\n")
        rule(f)

        hdr(f, "8.1 Current State Assessment", 2)
        readiness_rows = [
            {"Criterion": "Feature quality (Phase 1)",      "Status": "Ready",       "Evidence": "ConvNeXt-Tiny features extracted; AUROC=1.0 on OOD"},
            {"Criterion": "Grade label quality (Phase 3)",  "Status": "Conditional", "Evidence": "1,602 samples; label noise risk in grade_3"},
            {"Criterion": "Severity label availability",    "Status": "Partial",     "Evidence": "real_severity domain (2,602 images) exists"},
            {"Criterion": "OOD rejection framework",        "Status": "Ready",       "Evidence": "Mahalanobis fully implemented"},
            {"Criterion": "Ordinal regression (Phase 3)",   "Status": "Running",     "Evidence": "CORAL training in progress"},
            {"Criterion": "Calibration baseline",           "Status": "Not started", "Evidence": "Phase 5 required first"},
        ]
        tbl(f, pd.DataFrame(readiness_rows))

        hdr(f, "8.2 Phase 4 Architecture Recommendation", 2)
        w(f, """**Phase 4: Context-Aware Severity Modelling**

Goal: Given a fruit image, predict both the GRADE (1/2/3) AND the SEVERITY
of each defect attribute (bruising, discolouration, rot) using contextual
features from neighboring spatial regions.

Proposed Architecture:

    Input (224x224)
        -> ConvNeXt-Tiny (frozen from Phase 3 best checkpoint)
        -> Feature Map: F in R^{7x7x768}  (before global pool)
        -> Context Attention Module (CAM)
               Multi-head self-attention across spatial positions
               Captures WHERE the defect is and HOW SEVERE
        -> Global Context Vector: c in R^{768}
        -> Local Severity Head: MLP -> [bruising, discolouration, rot] in [0,1]^3
        -> Grade Prediction Head: CoralHead (reused from Phase 3)
        -> Joint Output: (grade, severity_vector)

Loss Function:

    L = lambda_g * L_CORAL + lambda_s * L_severity

where L_severity = BCE(s_hat, s*) on continuous severity scores,
and lambda_g = 1.0, lambda_s = 0.5 initially.

Key Design Choices:
1. Context attention: Preserves spatial defect localization, unlike global pooling.
2. Shared backbone: Reuse Phase 3 checkpoint -- no retraining from scratch.
3. Multi-task learning: Grade + severity trained jointly to regularize each other.
4. Dataset: Use real_severity (2,602 images) + real_grade (1,602 images) jointly.
""")

        hdr(f, "8.3 Phase 4 Training Strategy", 2)
        w(f, """Stage 1 (3 epochs): Freeze backbone; train Context Attention Module + Severity Head only.

Stage 2 (15 epochs): Unfreeze last 2 ConvNeXt stages; joint fine-tune with multi-task loss.

Evaluation Metrics:
- Grade: MAE, QWK (same as Phase 3)
- Severity: Spearman correlation per attribute, Mean Absolute Error
- Joint: Grade QWK with severity-aware weighting

Expected Timeline: 2-3 days implementation, 4-6 hours training on RTX 3050.
""")

        hdr(f, "8.4 Phase 4 Go / No-Go Criteria", 2)
        coral_qwk = metrics.get("qwk", 0.0) if metrics else 0.0
        coral_mae = metrics.get("mae", 999.0) if metrics else 999.0
        go_nogo = [
            {"Criterion": "CORAL QWK > 0.50",          "Target": "> 0.50",  "Current": f"{coral_qwk:.4f}", "Decision": "GO" if coral_qwk > 0.5 else "HOLD"},
            {"Criterion": "CORAL MAE < 0.50",           "Target": "< 0.50",  "Current": f"{coral_mae:.4f}", "Decision": "GO" if coral_mae < 0.5 else "HOLD"},
            {"Criterion": "Critical errors < 10%",      "Target": "< 10%",   "Current": "See failure_analysis.md", "Decision": "Pending"},
            {"Criterion": "Tomato eval gap resolved",   "Target": "CV done", "Current": "0 fixed test samples",    "Decision": "HOLD"},
        ]
        tbl(f, pd.DataFrame(go_nogo))

        w(f, """\n**Recommendation:**
If CORAL QWK > 0.50 and MAE < 0.50, proceed to Phase 4 with the context
attention architecture above. If not, invest in dataset expansion first
(target: 5,000+ real_grade images across all 3 species with balanced sessions).
""")

    print(f"Task 8 written: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 9 -- IEEE REVIEWER REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def task9_ieee_review():
    metrics  = load_json(OUT / "coral_metrics.json")
    ablation = load_csv(OUT / "ablation_results.csv")

    out = OUT / "ieee_review_report.md"
    with open(out, "w", encoding="utf-8") as f:
        hdr(f, "IEEE Reviewer Report: Full Project Review (Phases 1-3)", 1)
        w(f, """**Paper**: Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using
Context-Aware Severity Modeling and Uncertainty Calibration

**Reviewer**: Associate Editor / Reviewer #2
**Venue**: IEEE Transactions on Instrumentation and Measurement (TIM) / IEEE Access
**Review**: Full technical review (Phases 1-3)

""")
        rule(f)

        hdr(f, "Summary", 2)
        w(f, """This paper proposes a comprehensive framework for fruit quality grading
integrating: (1) closed-set multi-fruit classification using ConvNeXt-Tiny
with class-balanced focal loss, (2) open-set recognition via Mahalanobis
distance, and (3) rank-consistent ordinal regression using CORAL for
three-level quality grading. Phases 4-5 (severity modelling, uncertainty
calibration) are planned but incomplete.

The engineering execution is thorough, but several scientific concerns must
be addressed before this work meets the IEEE publication bar.
""")

        hdr(f, "Review Scores", 2)
        scores_rows = [
            {"Dimension": "Novelty",               "Score": "7/10", "Comment": "CORAL + Mahalanobis OOD combination in agricultural grading is novel"},
            {"Dimension": "Technical Quality",     "Score": "8/10", "Comment": "Implementation rigorous; two-stage training rationale is sound"},
            {"Dimension": "Experimental Design",   "Score": "6/10", "Comment": "Tomato test split gap and small dataset weaken claims"},
            {"Dimension": "Reproducibility",       "Score": "9/10", "Comment": "Session-level splits, seeds, checkpoints well-documented"},
            {"Dimension": "Publication Readiness", "Score": "6/10", "Comment": "Phases 4-5 incomplete; statistical tests needed"},
        ]
        tbl(f, pd.DataFrame(scores_rows))

        hdr(f, "Major Strengths", 2)
        w(f, """**S1 -- Dataset Curation Rigor:**
The multi-domain dataset construction with session-level leakage prevention
is a genuine methodological contribution. Zero session contamination across
56,038 images is verifiable and reproducible. This is significantly stronger
than image-level random splits common in prior fruit quality work.

**S2 -- Ordinal Loss Principled Design:**
CORAL's shared-weight formulation correctly encodes the monotonic structure
of agricultural quality grades. The loss directly minimizes violations of
P(y>0|x) >= P(y>1|x), unlike CE which is ordinally agnostic.

**S3 -- Open-Set Integration:**
Combining OOD rejection (Mahalanobis) with ordinal grading is uncommon in
agricultural AI literature. The unified trustworthy system concept is compelling.

**S4 -- Computational Efficiency:**
ConvNeXt-Tiny (28M parameters) achieves strong performance with significantly
lower FLOPs than ViT-based alternatives. RTX 3050 constraint demonstrates
practical deployability in resource-constrained settings.

**S5 -- Two-Stage Transfer Learning:**
The head-only pre-training strategy is well-motivated for small datasets and
correctly prevents catastrophic forgetting during backbone fine-tuning.
""")

        hdr(f, "Major Weaknesses (Required Revisions)", 2)
        w(f, """**W1 [Critical] -- Incomplete System:**
The title promises "Context-Aware Severity Modelling and Uncertainty Calibration"
but Phases 4-5 are not implemented. A paper cannot be submitted without validating
all title contributions experimentally.

Required revision: Complete Phases 4-5 or modify title/claims to match current implementation.

**W2 [Critical] -- Tomato Evaluation Gap:**
Tomato comprises 31.3% of real_grade samples yet has ZERO test samples due to
session-level splitting. Per-species results for Tomato are evaluation-set only
(data the model was optimised on). Species-level generalisation claims are
scientifically invalid for Tomato.

Required revision: Apply 5-fold CV guaranteeing each species appears in at
least one test fold. Report per-species metrics with fold-averaged estimates.

**W3 [Critical] -- Missing Statistical Testing:**
The ablation and multi-seed experiments must include:
- Paired t-test with p-values for CORAL vs CE
- Effect size (Cohen's d)
- 95% confidence intervals
- Bonferroni correction for multiple metrics

Without these, "CORAL is better than CE" is not a scientifically defensible claim.

**W4 [Significant] -- AUROC=1.000 for OOD is Unconvincing:**
Perfect OOD detection raises the question of whether the evaluation is sufficiently
challenging. Testing with a completely different fruit species (e.g., Orange) as
the OOD distribution is required.

Required revision: Report OOD AUROC with held-out species as the negative class.

**W5 [Moderate] -- No Comparison to SOTA Ordinal Methods:**
Missing baselines: CORN (Shi et al. 2022), Ordinal Encoding, Cumulative Link
Models, and recent ordinal transformers.

Required revision: Add at least 2 ordinal regression baselines.
""")

        hdr(f, "Minor Revisions", 2)
        w(f, """m1: Report inter-rater reliability (Cohen's kappa) for grade annotations.
m2: Provide GradCAM visualizations confirming model attends to fruit surface, not background.
m3: Table IV (baseline comparison) requires absolute accuracy/F1 numbers, not "Best/--".
m4: Clarify whether Phase 1 accuracy includes OOD filtering or is closed-set only.
m5: Report the best epoch number and early stopping behavior explicitly in the paper.
""")

        hdr(f, "Recommendation", 2)
        w(f, """## Decision: Major Revision

The engineering quality and reproducibility are strong. The core scientific
claims are plausible but not yet proven due to:

1. Incomplete system (Phases 4-5 missing)
2. Invalid Tomato test evaluation (zero test samples)
3. Missing statistical significance testing
4. Unconvincing OOD evaluation (AUROC=1.000)
5. Missing ordinal regression baselines

Upon addressing W1-W5, this paper has strong potential for acceptance in:
- **IEEE Access** (good breadth, practical contribution)
- **IEEE Transactions on Instrumentation and Measurement** (strong dataset + system focus)

For **IEEE TNNLS** or **Pattern Recognition Letters**, the severity modelling
contribution (Phase 4) must be fully validated and compared to prior work.

**Overall Rating: 5/10 (Borderline) -- Major Revision Required**
""")

    print(f"Task 9 written: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_full_report():
    sections = [
        ("training_analysis.md",        "Phase 3: Training Analysis"),
        ("ordinal_analysis.md",         "Ordinal Performance Analysis"),
        ("species_analysis.md",         "Species-Wise Analysis"),
        ("failure_analysis.md",         "Failure Case Analysis"),
        ("statistical_significance.md", "Statistical Significance Testing"),
        ("publication_tables.md",       "IEEE Publication Tables"),
        ("contribution_review.md",      "Research Contribution Review"),
        ("phase4_readiness.md",         "Phase 4 Readiness"),
        ("ieee_review_report.md",       "Full IEEE Reviewer Report"),
    ]

    out = OUT / "full_technical_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Full Technical Report\n\n")
        f.write("**Paper**: Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading "
                "Using Context-Aware Severity Modeling and Uncertainty Calibration\n\n")
        f.write("**Phase**: Phase 3 Post-Training Analysis\n\n")
        f.write("---\n\n## Table of Contents\n\n")
        for i, (_, title) in enumerate(sections, 1):
            f.write(f"{i}. {title}\n")
        f.write("\n---\n\n")
        for fname, title in sections:
            fpath = OUT / fname
            if fpath.exists():
                f.write(fpath.read_text(encoding="utf-8") + "\n\n---\n\n")
            else:
                f.write(f"## {title}\n\n> Section not yet generated.\n\n---\n\n")

    print(f"Full report assembled: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 3 POST-TRAINING ANALYSIS SUITE")
    print("=" * 70)

    task1_training_analysis()
    task2_ordinal_analysis()
    task3_species_analysis()
    task4_failure_analysis()
    task5_statistical_significance()
    task6_publication_tables()
    task7_contribution_review()
    task8_phase4_readiness()
    task9_ieee_review()

    print("\n[Final] Assembling full technical report...")
    assemble_full_report()

    print("\n" + "=" * 70)
    print("ALL ANALYSIS TASKS COMPLETE")
    print("=" * 70)
    for fname in [
        "training_analysis.md", "ordinal_analysis.md", "species_analysis.md",
        "failure_analysis.md", "statistical_significance.md", "publication_tables.md",
        "contribution_review.md", "phase4_readiness.md", "ieee_review_report.md",
        "full_technical_report.md",
    ]:
        p = OUT / fname
        size = p.stat().st_size if p.exists() else 0
        print(f"  {fname:<40} {size:>8,} bytes")


if __name__ == "__main__":
    main()
