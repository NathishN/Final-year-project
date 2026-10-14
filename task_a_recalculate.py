"""
Phase 4.4 Task A: Independent recalculation of all 20 paired results.
Reads raw CSV, verifies completeness, recomputes all statistics from scratch.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

os.makedirs("outputs/casm_v2", exist_ok=True)

# --- Load raw CSV ---
df = pd.read_csv("casmv2_validation_results.csv")
print(f"Raw CSV rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Model values: {df.Model.unique().tolist()}")
print()

base_df = df[df.Model == "Baseline"].copy().sort_values("Seed").reset_index(drop=True)
v2_df   = df[df.Model == "CASMv2"].copy().sort_values("Seed").reset_index(drop=True)

# --- Verify seed completeness ---
base_seeds = sorted(base_df.Seed.tolist())
v2_seeds   = sorted(v2_df.Seed.tolist())
print(f"Baseline seeds ({len(base_seeds)}): {base_seeds}")
print(f"CASMv2   seeds ({len(v2_seeds)}): {v2_seeds}")
assert base_seeds == v2_seeds, f"SEED MISMATCH: {set(base_seeds)^set(v2_seeds)}"
print("Seed pairing: OK - all seeds present in both models")
N = len(base_seeds)
print()

# --- Verify paired alignment ---
assert list(base_df.Seed) == list(v2_df.Seed), "Seed order mismatch after sort"

# --- Per-seed differences ---
metrics = ["QWK","MAE","Accuracy","OBO_Acc","Macro_F1"]
diff_rows = []
for i, seed in enumerate(base_seeds):
    b = base_df.iloc[i]
    v = v2_df.iloc[i]
    row = {"Seed": seed}
    for m in metrics:
        bval = float(b[m])
        vval = float(v[m])
        diff = vval - bval
        # Better = higher for all metrics EXCEPT MAE (lower is better)
        better = (diff > 0) if m != "MAE" else (diff < 0)
        row[f"Baseline_{m}"]  = round(bval, 6)
        row[f"CASMv2_{m}"]    = round(vval, 6)
        row[f"Diff_{m}"]      = round(diff, 6)
        row[f"Better_{m}"]    = better
    diff_rows.append(row)
df_diff = pd.DataFrame(diff_rows)
df_diff.to_csv("outputs/casm_v2/phase44_per_seed_differences.csv", index=False)
print("Saved phase44_per_seed_differences.csv")

# --- Recompute aggregated stats ---
rec_rows = []
for model_label, sub in [("Baseline", base_df), ("CASMv2", v2_df)]:
    for m in metrics:
        vals = sub[m].values.astype(float)
        rec_rows.append({
            "Model": model_label,
            "Metric": m,
            "N": N,
            "Mean":   round(float(np.mean(vals)), 6),
            "Std":    round(float(np.std(vals, ddof=1)), 6),
            "Median": round(float(np.median(vals)), 6),
            "Min":    round(float(np.min(vals)), 6),
            "Max":    round(float(np.max(vals)), 6),
        })
df_rec = pd.DataFrame(rec_rows)
df_rec.to_csv("outputs/casm_v2/phase44_recalculated_metrics.csv", index=False)
print("Saved phase44_recalculated_metrics.csv")
print()

# --- Full statistical analysis per metric ---
print("=== INDEPENDENT RECALCULATION ===")
print()

stat_rows = []
for m in metrics:
    bv = base_df[m].values.astype(float)
    cv = v2_df[m].values.astype(float)
    dv = cv - bv                      # paired differences: CASMv2 - Baseline

    # Sign conventions: positive diff = CASMv2 better for all except MAE
    # For MAE: negative diff = CASMv2 better (lower MAE)

    n_pos = int(np.sum(dv > 0))      # CASMv2 higher (good for all non-MAE)
    n_neg = int(np.sum(dv < 0))      # CASMv2 lower
    n_tie = int(np.sum(dv == 0))

    mean_diff  = float(np.mean(dv))
    std_diff   = float(np.std(dv, ddof=1))
    med_diff   = float(np.median(dv))
    se_diff    = std_diff / np.sqrt(N)
    ci95_lo    = mean_diff - 1.96 * se_diff   # approximate; exact below
    ci95_hi    = mean_diff + 1.96 * se_diff

    # Two-sided paired t-test (PRIMARY for each metric)
    t_stat, p_twosided = stats.ttest_rel(cv, bv)
    # Exact 95% CI from t-distribution
    ci95_lo_t  = mean_diff - stats.t.ppf(0.975, df=N-1) * se_diff
    ci95_hi_t  = mean_diff + stats.t.ppf(0.975, df=N-1) * se_diff

    # Two-sided Wilcoxon signed-rank (sensitivity)
    try:
        w_stat, p_wilcox_2s = stats.wilcoxon(dv, alternative="two-sided")
    except ValueError:
        w_stat, p_wilcox_2s = np.nan, np.nan

    # One-sided Wilcoxon (SECONDARY, exploratory only)
    try:
        _, p_wilcox_1s = stats.wilcoxon(dv, alternative="greater")
    except ValueError:
        p_wilcox_1s = np.nan

    # Cohen d for paired data: mean(diff) / std(diff)
    cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # Win/tie/loss
    if m != "MAE":
        wins   = n_pos
        losses = n_neg
    else:
        wins   = n_neg   # lower MAE = win for CASMv2
        losses = n_pos

    stat_rows.append({
        "Metric": m,
        "Mean_diff": round(mean_diff, 6),
        "Std_diff":  round(std_diff, 6),
        "Median_diff": round(med_diff, 6),
        "CI95_lo":   round(ci95_lo_t, 6),
        "CI95_hi":   round(ci95_hi_t, 6),
        "t_stat":    round(float(t_stat), 4),
        "df":        N-1,
        "p_twosided_t": round(float(p_twosided), 4),
        "W_stat":    float(w_stat) if not np.isnan(w_stat) else np.nan,
        "p_2sided_wilcoxon": round(float(p_wilcox_2s), 4) if not np.isnan(p_wilcox_2s) else np.nan,
        "p_1sided_wilcoxon": round(float(p_wilcox_1s), 4) if not np.isnan(p_wilcox_1s) else np.nan,
        "Cohen_d":   round(cohen_d, 4),
        "Wins_v2":   wins,
        "Losses_v2": losses,
        "Ties":      n_tie,
    })

    print(f"--- {m} ---")
    print(f"  Baseline: mean={np.mean(bv):.4f}  std={np.std(bv,ddof=1):.4f}  med={np.median(bv):.4f}")
    print(f"  CASMv2:   mean={np.mean(cv):.4f}  std={np.std(cv,ddof=1):.4f}  med={np.median(cv):.4f}")
    print(f"  Diff:     mean={mean_diff:+.4f}  std={std_diff:.4f}  med={med_diff:+.4f}")
    print(f"  95% CI:   [{ci95_lo_t:+.4f}, {ci95_hi_t:+.4f}]")
    print(f"  t-test (2-sided): t={t_stat:.4f}  p={p_twosided:.4f}")
    print(f"  Wilcoxon (2-sided): W={w_stat:.1f}  p={p_wilcox_2s:.4f}")
    print(f"  Wilcoxon (1-sided, secondary): p={p_wilcox_1s:.4f}")
    print(f"  Cohen d (paired): {cohen_d:+.4f}")
    print(f"  Wins/Ties/Losses (CASMv2): {wins}/{n_tie}/{losses}")
    print()

df_stat = pd.DataFrame(stat_rows)

# --- Holm-Bonferroni correction across all 5 metrics ---
print("=== HOLM-BONFERRONI CORRECTION (all 5 metrics as confirmatory family) ===")
raw_ps = df_stat["p_twosided_t"].values.astype(float)
order  = np.argsort(raw_ps)
adj_ps = np.zeros(len(raw_ps))
for rank, idx in enumerate(order):
    adj_ps[idx] = min(1.0, raw_ps[idx] * (len(raw_ps) - rank))
# Enforce monotonicity
for i in range(len(order)-1, 0, -1):
    adj_ps[order[i-1]] = min(adj_ps[order[i-1]], adj_ps[order[i]])

df_stat["p_holm_bonferroni"] = [round(p, 4) for p in adj_ps]
df_stat["sig_raw_005"]       = df_stat["p_twosided_t"] < 0.05
df_stat["sig_holm_005"]      = df_stat["p_holm_bonferroni"] < 0.05

for i, row in df_stat.iterrows():
    print(f"  {row.Metric}: raw_p={row.p_twosided_t:.4f}  holm_adj_p={row.p_holm_bonferroni:.4f}  sig_raw={row.sig_raw_005}  sig_holm={row.sig_holm_005}")

print()

# --- Bootstrap CI for QWK difference ---
print("=== BOOTSTRAP 95% CI FOR QWK DIFFERENCE ===")
print("  Protocol: matched-seed paired resampling, 10,000 reps, BCa not available => percentile method")
qwk_diffs = (v2_df["QWK"].values - base_df["QWK"].values).astype(float)
rng = np.random.default_rng(2024)
boot_means = np.array([
    np.mean(rng.choice(qwk_diffs, size=N, replace=True))
    for _ in range(10000)
])
boot_ci95_lo = float(np.percentile(boot_means, 2.5))
boot_ci95_hi = float(np.percentile(boot_means, 97.5))
boot_ci90_lo = float(np.percentile(boot_means, 5.0))
boot_ci90_hi = float(np.percentile(boot_means, 95.0))
print(f"  Bootstrap 95% CI (percentile): [{boot_ci95_lo:+.4f}, {boot_ci95_hi:+.4f}]")
print(f"  Bootstrap 90% CI (percentile): [{boot_ci90_lo:+.4f}, {boot_ci90_hi:+.4f}]")
print(f"  Resampling unit: matched paired seed-level differences")
print(f"  N reps: 10,000 | RNG seed: 2024 | N pairs: {N}")
print(f"  Limitation: N=20 pairs is at the lower boundary for stable percentile bootstrap CIs")
print()

# --- Compare with reported values ---
print("=== DISCREPANCY CHECK vs PREVIOUSLY REPORTED ===")
reported = {
    "QWK":      {"mean_base": 0.7032, "std_base": 0.0728, "mean_v2": 0.7326, "std_v2": 0.0195,
                 "delta": 0.0294, "p_t": 0.0593, "p_wilcox_1s": 0.0319, "cohen_d": 0.5520,
                 "wins": 13},
    "MAE":      {"p_t": 0.0468},
    "OBO_Acc":  {"p_t": 0.0266},
}
qwk_row = df_stat[df_stat.Metric == "QWK"].iloc[0]
print(f"QWK:")
print(f"  Reported mean_base={reported['QWK']['mean_base']:.4f}  Recalc={df_rec[(df_rec.Model=='Baseline')&(df_rec.Metric=='QWK')]['Mean'].values[0]:.4f}  match={abs(reported['QWK']['mean_base']-df_rec[(df_rec.Model=='Baseline')&(df_rec.Metric=='QWK')]['Mean'].values[0])<1e-3}")
print(f"  Reported std_base={reported['QWK']['std_base']:.4f}   Recalc={df_rec[(df_rec.Model=='Baseline')&(df_rec.Metric=='QWK')]['Std'].values[0]:.4f}")
print(f"  Reported p_t={reported['QWK']['p_t']:.4f}     Recalc={qwk_row.p_twosided_t:.4f}")
print(f"  Reported wins=13   Recalc={qwk_row.Wins_v2}")
print(f"  Reported p_wilcox_1s={reported['QWK']['p_wilcox_1s']:.4f}  Recalc_1s={qwk_row.p_1sided_wilcoxon:.4f}  Recalc_2s={qwk_row.p_2sided_wilcoxon:.4f}")
print()

df_stat.to_csv("outputs/casm_v2/phase44_statistical_report_data.csv", index=False)

# Save bootstrap results
boot_dict = {
    "qwk_diff_mean": float(np.mean(qwk_diffs)),
    "qwk_diff_std":  float(np.std(qwk_diffs, ddof=1)),
    "boot_95_lo": boot_ci95_lo, "boot_95_hi": boot_ci95_hi,
    "boot_90_lo": boot_ci90_lo, "boot_90_hi": boot_ci90_hi,
    "n_reps": 10000, "rng_seed": 2024, "n_pairs": N,
    "holm_bonferroni": {
        row.Metric: {"raw_p": row.p_twosided_t, "adj_p": row.p_holm_bonferroni,
                     "sig_raw": bool(row.sig_raw_005), "sig_holm": bool(row.sig_holm_005)}
        for _, row in df_stat.iterrows()
    }
}
import json
json.dump(boot_dict, open("outputs/casm_v2/phase44_bootstrap_data.json","w"), indent=2)
print("Saved all Task A outputs.")
print("TASK A COMPLETE")
