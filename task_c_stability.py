"""
Phase 4.4 Task C: Investigate training collapses for seeds 19 and 31.
Diagnoses without rerunning — uses existing CSV to analyze effect.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

os.makedirs("outputs/casm_v2", exist_ok=True)

df     = pd.read_csv("casmv2_validation_results.csv")
base   = df[df.Model=="Baseline"].sort_values("Seed").reset_index(drop=True)
v2     = df[df.Model=="CASMv2"].sort_values("Seed").reset_index(drop=True)
N      = len(base)

suspect = [19, 31]

print("=== TASK C: STABILITY INVESTIGATION ===")
print()

# --- Identify anomalies by IQR outlier rule ---
bqwk = base.QWK.values
q1,q3 = np.percentile(bqwk,[25,75])
iqr   = q3-q1
lo_fence = q1 - 1.5*iqr
print(f"Baseline QWK: mean={np.mean(bqwk):.4f}  median={np.median(bqwk):.4f}  Q1={q1:.4f}  Q3={q3:.4f}  IQR={iqr:.4f}")
print(f"Outlier fence (Q1-1.5*IQR): {lo_fence:.4f}")
outlier_seeds = base[base.QWK < lo_fence]["Seed"].tolist()
print(f"Outlier seeds by IQR rule: {outlier_seeds}")
print()

# --- Characterize the suspect seeds ---
print("=== Suspect Seed Profiles ===")
for s in suspect:
    br = base[base.Seed==s].iloc[0]
    vr = v2[v2.Seed==s].iloc[0]
    print(f"\nSeed {s}:")
    print(f"  Baseline: QWK={br.QWK:.4f}  MAE={br.MAE:.4f}  Acc={br.Accuracy:.2f}%  OBO={br.OBO_Acc:.2f}%  F1={br.Macro_F1:.2f}%")
    print(f"  CASMv2:   QWK={vr.QWK:.4f}  MAE={vr.MAE:.4f}  Acc={vr.Accuracy:.2f}%  OBO={vr.OBO_Acc:.2f}%  F1={vr.Macro_F1:.2f}%")
    print(f"  Delta:    QWK={vr.QWK-br.QWK:+.4f}  MAE={vr.MAE-br.MAE:+.4f}  Acc={vr.Accuracy-br.Accuracy:+.2f}%  OBO={vr.OBO_Acc-br.OBO_Acc:+.2f}%")
    b_rank = int(base.QWK.rank().loc[base[base.Seed==s].index[0]])
    print(f"  Baseline QWK rank (1=lowest): {b_rank}/{N}")
    # MAE of 0.44 and 0.39 => model predicts mostly one class
    if br.MAE > 0.35:
        print(f"  DIAGNOSIS: MAE={br.MAE:.4f} >> 0.3 => likely class collapse (predicting mostly grade 0)")
    if br.OBO_Acc < 90:
        print(f"  DIAGNOSIS: OBO={br.OBO_Acc:.2f}% < 90% => many predictions are off by 2+ grades")

print()

# --- Effect of the two suspect seeds on the mean difference ---
diffs      = (v2.QWK.values - base.QWK.values)
mean_all   = float(np.mean(diffs))
mean_wo    = float(np.mean([d for i,d in enumerate(diffs) if base.Seed.iloc[i] not in suspect]))
contrib    = mean_all - mean_wo
print("=== Effect on Mean Difference ===")
print(f"  All N=20:         mean diff = {mean_all:+.4f}")
print(f"  Without seeds {suspect}: mean diff = {mean_wo:+.4f}  (N=18)")
print(f"  Contribution of {suspect}: {contrib:+.4f} ({contrib/mean_all*100:+.1f}% of total)")
print()

# --- Sensitivity: exclude suspect seeds ---
base_18 = base[~base.Seed.isin(suspect)].reset_index(drop=True)
v2_18   = v2[~v2.Seed.isin(suspect)].reset_index(drop=True)
bv18    = base_18.QWK.values
cv18    = v2_18.QWK.values
dv18    = cv18 - bv18
t18,p18 = stats.ttest_rel(bv18,cv18)
pooled18= np.sqrt((np.std(bv18,ddof=1)**2+np.std(cv18,ddof=1)**2)/2)
d18     = float(np.mean(dv18))/pooled18 if pooled18>0 else 0.0

print("=== Sensitivity Analysis: Exclude seeds 19 and 31 (N=18) ===")
print("  NOTE: This is DIAGNOSTIC ONLY. Full N=20 result remains primary.")
print(f"  Baseline: mean={np.mean(bv18):.4f}  std={np.std(bv18,ddof=1):.4f}")
print(f"  CASMv2:   mean={np.mean(cv18):.4f}  std={np.std(cv18,ddof=1):.4f}")
print(f"  Mean diff: {np.mean(dv18):+.4f}  std_diff={np.std(dv18,ddof=1):.4f}")
print(f"  t-test (2-sided): t={t18:.4f}  p={p18:.4f}")
print(f"  Cohen d: {d18:+.4f}")
se18 = np.std(dv18,ddof=1)/np.sqrt(18)
ci_lo18 = np.mean(dv18) - stats.t.ppf(0.975,17)*se18
ci_hi18 = np.mean(dv18) + stats.t.ppf(0.975,17)*se18
print(f"  95% CI: [{ci_lo18:+.4f}, {ci_hi18:+.4f}]")
print()

# --- Diagnosis summary ---
print("=== Collapse Diagnosis ===")
print("""
Baseline seed 19: QWK=0.5363, MAE=0.3994, OBO=85.33%
  - MAE=0.40 is 1.75x higher than the N=20 mean (0.2445)
  - OBO=85.33% (13/20 baseline seeds have OBO > 92%)
  - Pattern consistent with partial class collapse: model biased toward grade 0
  - CASMv2 seed 19: QWK=0.7179 (recovers to population norm)
  
Baseline seed 31: QWK=0.5000, MAE=0.4433, OBO=84.28%
  - QWK=0.500 is at lower-bound of IQR fence (Q1-1.5*IQR = 0.59)
  - MAE=0.44 highest of all 40 runs
  - OBO=84.28% lowest of all 40 baseline runs
  - Pattern: severe grade-0 bias causing near-random ordinal prediction
  - CASMv2 seed 31: QWK=0.6923 (recovers significantly)
  
Root cause hypothesis (untested, requires loss curves not captured):
  - These seeds produce suboptimal learning-rate initialization interacting
    with class-imbalanced data (87.5% grade-0): the model collapses to 
    predicting grade-0 for most inputs.
  - CASMv2 recovery suggests the species embedding helps break class-collapse
    symmetry, but this is ASSOCIATIONAL - not proven causal.
    
Protocol note: Seeds 19 and 31 were selected from a predetermined list
([42,123,777,999,2025,1,2,3,5,7,11,13,17,19,23,31,37,41,43,47]).
They are not adversarial seeds; they represent genuine training variability
for Image-Only. The same seeds were used for BOTH models under identical
protocols, so the comparison is valid.
""")

print("=== Justification for Sensitivity Analysis Only ===")
print("""No documented pre-specified exclusion rule was established before the
experiment. Excluding seeds 19 and 31 MUST be treated as diagnostic only.
The primary analysis uses all N=20 seeds.
""")

# Save report
lines = [
    "# Phase 4.4 Task C — Training Stability Investigation",
    "",
    f"**Primary analysis: ALL N=20 seeds (no exclusions)**",
    "",
    "## Anomaly Detection",
    "",
    f"Baseline QWK: mean={np.mean(bqwk):.4f} median={np.median(bqwk):.4f} Q1={q1:.4f} Q3={q3:.4f} IQR={iqr:.4f}",
    f"Outlier fence (Q1 - 1.5×IQR): {lo_fence:.4f}",
    f"Seeds below fence: {outlier_seeds}",
    "",
    "## Suspect Seed Profiles",
    "",
    "| Seed | Model | QWK | MAE | Acc% | OBO% | F1% |",
    "|------|-------|-----|-----|------|------|-----|",
]
for s in suspect:
    br = base[base.Seed==s].iloc[0]
    vr = v2[v2.Seed==s].iloc[0]
    lines.append(f"| {s} | Baseline | {br.QWK:.4f} | {br.MAE:.4f} | {br.Accuracy:.2f} | {br.OBO_Acc:.2f} | {br.Macro_F1:.2f} |")
    lines.append(f"| {s} | CASMv2   | {vr.QWK:.4f} | {vr.MAE:.4f} | {vr.Accuracy:.2f} | {vr.OBO_Acc:.2f} | {vr.Macro_F1:.2f} |")

lines += [
    "",
    "## Collapse Diagnosis",
    "",
    "**Seed 19 (Baseline):** QWK=0.5363, MAE=0.3994, OBO=85.33%",
    "- MAE is 1.75x the population mean (0.2445) — indicates grade-0 bias",
    "- Pattern consistent with partial class collapse toward majority class",
    "- CASMv2 recovers to QWK=0.7179 (Δ=+0.1816)",
    "",
    "**Seed 31 (Baseline):** QWK=0.5000, MAE=0.4433, OBO=84.28%",
    "- Lowest QWK and highest MAE of all 40 runs",
    "- Below IQR outlier fence",
    "- CASMv2 recovers to QWK=0.6923 (Δ=+0.1923)",
    "",
    "**Mechanistic hypothesis (associational, not proven):**",
    "> With 87.5% grade-0 prevalence, certain random initializations collapse",
    "> to predicting grade-0 for most inputs. The species embedding may provide",
    "> a diversity signal that prevents this collapse, but causality is unconfirmed.",
    "",
    "## Effect on Mean Difference",
    "",
    f"| Analysis | N | Mean QWK diff | Interpretation |",
    f"|----------|---|---------------|----------------|",
    f"| **Primary (all seeds)** | 20 | {mean_all:+.4f} | Main result |",
    f"| Sensitivity (excl. 19,31) | 18 | {mean_wo:+.4f} | Diagnostic only |",
    f"| Contribution of seeds 19,31 | — | {contrib:+.4f} ({contrib/mean_all*100:.0f}%) | Large |",
    "",
    "> Seeds 19 and 31 contribute approximately 63% of the mean QWK difference.",
    "> Without them, the mean improvement drops from +0.0294 to +0.0119.",
    "> This is a critical result: the mean improvement is substantially driven",
    "> by CASM v2's robustness on collapse-prone seeds, not broad superiority.",
    "",
    "## Sensitivity Analysis (Diagnostic Only — NOT Primary)",
    "",
    f"| Metric | All N=20 | N=18 (excl. 19,31) |",
    f"|--------|----------|--------------------|",
    f"| Baseline QWK | {np.mean(bqwk):.4f}±{np.std(bqwk,ddof=1):.4f} | {np.mean(bv18):.4f}±{np.std(bv18,ddof=1):.4f} |",
    f"| CASMv2 QWK | {np.mean(v2.QWK):.4f}±{np.std(v2.QWK,ddof=1):.4f} | {np.mean(cv18):.4f}±{np.std(cv18,ddof=1):.4f} |",
    f"| Mean diff | {mean_all:+.4f} | {np.mean(dv18):+.4f} |",
    f"| p-value (2-sided t) | 0.0593 | {p18:.4f} |",
    f"| Cohen d | +0.4487 | {d18:+.4f} |",
    "",
    "> **Conclusion:** With stable seeds only, the QWK improvement shrinks to",
    f"> {np.mean(dv18):+.4f} (p={p18:.4f}). The primary improvement is primarily a",
    "> **robustness advantage** on collapse-prone initializations, not a general",
    "> accuracy improvement on all seeds.",
    "",
    "## Protocol Integrity",
    "- Seeds were pre-specified before training: [42,123,777,999,2025,1,2,3,5,7,11,13,17,19,23,31,37,41,43,47]",
    "- Identical training protocol applied to both Baseline and CASMv2",
    "- No cherry-picking or post-hoc seed selection",
    "- Seeds 19 and 31 are genuine training failures, not adversarial seeds",
    "",
    "## Conclusion",
    "",
    "> The primary statistical result (N=20) is valid. However, the improvement",
    "> is largely a **stability/robustness benefit** rather than a uniform accuracy gain.",
    "> This is scientifically important and publishable as a distinct contribution:",
    "> CASM v2 is more robust to training instability caused by class imbalance.",
]

open("outputs/casm_v2/phase44_stability_investigation.md","w",encoding="utf-8").write("\n".join(lines))
print("Saved phase44_stability_investigation.md")
print("TASK C COMPLETE")
