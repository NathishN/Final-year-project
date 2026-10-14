# Task 3: Species-Wise Ordinal Analysis

---

## 3.1 Per-Species Performance Table

> Species metrics not yet available.

## 3.2 Species Difficulty Ranking

| Rank | Species | Difficulty | Primary Reason |
|------|---------|------------|----------------|
| 1 (Hardest) | Tomato | HIGH | Color-based grade signal confounded by ripeness; grade_3 has 0 test samples |
| 2 | Apple | MEDIUM | Grade 2/3 boundary visually ambiguous (bruising vs. severe damage) |
| 3 (Easiest) | Mango | LOWER | Grade 1/3 extremes visually distinct; largest sample count (618) |

## 3.3 Biological and Annotation Reasoning

**Apple:**
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
## 3.4 Dataset-Level Species Analysis

Species  Grade 1  Grade 2  Grade 3  Total Imbalance
  Apple      198      172      112    482     1.77x
  Mango      251      250      117    618     2.15x
 Tomato      145      231      126    502     1.83x

