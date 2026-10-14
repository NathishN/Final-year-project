import os
import json
import numpy as np
import pandas as pd
from scipy import stats

print("=== Phase 4.5 Task 4: Statistical Reproduction & Reconciliation ===")

# Load raw ablation CSV (80 runs: 4 configs x 20 seeds)
raw_abl_path = "outputs/casm_v2/phase44_ablation_raw.csv"
df_abl = pd.read_csv(raw_abl_path)
print(f"Loaded {raw_abl_path}: {len(df_abl)} rows across configs {df_abl['Config'].unique().tolist()}")

# Load casmv2 validation results (40 runs: Baseline vs CASMv2)
raw_val_path = "casmv2_validation_results.csv"
df_val = pd.read_csv(raw_val_path)
print(f"Loaded {raw_val_path}: {len(df_val)} rows across models {df_val['Model'].unique().tolist()}")

METRICS = ["QWK", "MAE", "Accuracy", "OBO_Acc", "Macro_F1"]

# -------------------------------------------------------------------------
# 1. Metric Reproduction: Summary statistics per Configuration
# -------------------------------------------------------------------------
metric_reprod_rows = []
for cfg in ["A", "B", "C", "D"]:
    sub = df_abl[df_abl["Config"] == cfg]
    name = sub["Name"].iloc[0]
    row = {"Config": cfg, "Name": name, "N_seeds": len(sub)}
    for m in METRICS:
        vals = sub[m].values.astype(float)
        row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
        row[f"{m}_std"]  = round(float(np.std(vals, ddof=1)), 4)
        row[f"{m}_median"] = round(float(np.median(vals)), 4)
        row[f"{m}_min"] = round(float(np.min(vals)), 4)
        row[f"{m}_max"] = round(float(np.max(vals)), 4)
    metric_reprod_rows.append(row)

df_metric_reprod = pd.DataFrame(metric_reprod_rows)
df_metric_reprod.to_csv("outputs/casm_v2/phase45_metric_reproduction.csv", index=False)
print("Saved outputs/casm_v2/phase45_metric_reproduction.csv")

# -------------------------------------------------------------------------
# 2. Paired 20-Seed Comparison: Baseline (A) vs CASM v2 (D)
# -------------------------------------------------------------------------
sub_a = df_abl[df_abl["Config"] == "A"].sort_values("Seed").reset_index(drop=True)
sub_d = df_abl[df_abl["Config"] == "D"].sort_values("Seed").reset_index(drop=True)
assert list(sub_a["Seed"]) == list(sub_d["Seed"]), "Seed alignment error between A and D!"
N = len(sub_a)

paired_stats_rows = []
for m in METRICS:
    va = sub_a[m].values.astype(float)
    vd = sub_d[m].values.astype(float)
    diff = vd - va # CASM v2 (D) - Baseline (A)
    
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se_diff = std_diff / np.sqrt(N)
    
    # Paired Student's t-test (two-sided)
    t_stat, p_t_2s = stats.ttest_rel(vd, va)
    
    # Wilcoxon signed-rank test (two-sided)
    w_stat, p_w_2s = stats.wilcoxon(diff, alternative="two-sided")
    
    # Paired effect size: Cohen's d_z = mean(diff) / std(diff)
    cohen_dz = mean_diff / std_diff if std_diff > 0 else 0.0
    
    # Independent pooled effect size for comparison: Cohen's d_s
    s_pooled = np.sqrt((np.std(va, ddof=1)**2 + np.std(vd, ddof=1)**2) / 2.0)
    cohen_ds = mean_diff / s_pooled if s_pooled > 0 else 0.0
    
    # 95% Confidence Interval from t-distribution
    ci_t_lo = mean_diff - stats.t.ppf(0.975, df=N-1) * se_diff
    ci_t_hi = mean_diff + stats.t.ppf(0.975, df=N-1) * se_diff
    
    # Bootstrap 95% CI (10,000 resamples of paired differences)
    rng = np.random.default_rng(2024)
    boot_means = [np.mean(rng.choice(diff, size=N, replace=True)) for _ in range(10000)]
    ci_boot_lo = float(np.percentile(boot_means, 2.5))
    ci_boot_hi = float(np.percentile(boot_means, 97.5))
    
    # Win / tie / loss count (D vs A)
    better_d = (diff > 0) if m != "MAE" else (diff < 0)
    wins = int(np.sum(better_d))
    ties = int(np.sum(diff == 0))
    losses = N - wins - ties
    
    paired_stats_rows.append({
        "Metric": m,
        "Baseline_mean": round(float(np.mean(va)), 4),
        "CASMv2_mean": round(float(np.mean(vd)), 4),
        "Mean_diff": round(mean_diff, 4),
        "Std_diff": round(std_diff, 4),
        "SE_diff": round(se_diff, 4),
        "CI95_t_lo": round(ci_t_lo, 4),
        "CI95_t_hi": round(ci_t_hi, 4),
        "CI95_boot_lo": round(ci_boot_lo, 4),
        "CI95_boot_hi": round(ci_boot_hi, 4),
        "t_stat": round(float(t_stat), 4),
        "df": N - 1,
        "p_twosided_t": round(float(p_t_2s), 4),
        "W_stat": round(float(w_stat), 1),
        "p_twosided_wilcoxon": round(float(p_w_2s), 4),
        "Cohen_dz_paired": round(cohen_dz, 4),
        "Cohen_ds_pooled": round(cohen_ds, 4),
        "Wins_v2": wins,
        "Ties": ties,
        "Losses_v2": losses
    })

df_paired = pd.DataFrame(paired_stats_rows)

# Apply Holm-Bonferroni across the family of 5 metrics
raw_ps = df_paired["p_twosided_t"].values
order = np.argsort(raw_ps)
adj_ps = np.zeros(len(raw_ps))
for rank, idx in enumerate(order):
    adj_ps[idx] = min(1.0, raw_ps[idx] * (len(raw_ps) - rank))
for i in range(len(order) - 1, 0, -1):
    adj_ps[order[i-1]] = min(adj_ps[order[i-1]], adj_ps[order[i]])

df_paired["p_holm_bonferroni"] = [round(p, 4) for p in adj_ps]
df_paired["sig_raw_005"] = df_paired["p_twosided_t"] < 0.05
df_paired["sig_holm_005"] = df_paired["p_holm_bonferroni"] < 0.05

df_paired.to_csv("outputs/casm_v2/phase45_paired_statistics.csv", index=False)
print("Saved outputs/casm_v2/phase45_paired_statistics.csv")

# -------------------------------------------------------------------------
# 2b. Sensitivity Analysis (N=18 seeds, excluding Seeds 19 and 31)
# -------------------------------------------------------------------------
mask_18 = ~sub_a["Seed"].isin([19, 31])
va_18 = sub_a.loc[mask_18, "QWK"].values.astype(float)
vd_18 = sub_d.loc[mask_18, "QWK"].values.astype(float)
diff_18 = vd_18 - va_18
t_18, p_18 = stats.ttest_rel(vd_18, va_18)
w_18, pw_18 = stats.wilcoxon(diff_18, alternative="two-sided")
mean_d_18 = float(np.mean(diff_18))
std_d_18 = float(np.std(diff_18, ddof=1))
dz_18 = mean_d_18 / std_d_18 if std_d_18 > 0 else 0.0

print("\n--- Sensitivity Analysis: N=18 Seeds (Excluding Outliers 19 & 31) ---")
print(f"  Baseline QWK mean: {np.mean(va_18):.4f} +/- {np.std(va_18, ddof=1):.4f}")
print(f"  CASM v2 QWK mean:  {np.mean(vd_18):.4f} +/- {np.std(vd_18, ddof=1):.4f}")
print(f"  Mean Difference:   {mean_d_18:+.4f} (std={std_d_18:.4f})")
print(f"  Paired t-test:     t(17) = {t_18:.4f}, p = {p_18:.4f}")
print(f"  Wilcoxon test:     W = {w_18:.1f}, p = {pw_18:.4f}")
print(f"  Cohen's d_z:       {dz_18:+.4f}")

# -------------------------------------------------------------------------
# 3. Factorial Ablation Contrasts Reconciliation (Task 4.3)
# -------------------------------------------------------------------------
# Contrasts to evaluate:
# 1. D vs A: Species+OOD vs Image-Only (Primary)
# 2. B vs A: Species vs Image-Only (Isolates Species on Image)
# 3. C vs A: OOD vs Image-Only (Isolates OOD on Image)
# 4. D vs B: Species+OOD vs Species (Isolates OOD addition to Species)
# 5. D vs C: Species+OOD vs OOD (Isolates Species addition to OOD)
# 6. B vs D: Species vs Species+OOD (Direct test of whether Species alone beats full model)

contrasts_spec = [
    ("D", "A", "Primary: Species+OOD (D) vs Image-Only (A)"),
    ("B", "A", "Secondary: Species (B) vs Image-Only (A)"),
    ("C", "A", "Secondary: OOD (C) vs Image-Only (A)"),
    ("D", "B", "Secondary: Species+OOD (D) vs Species (B)"),
    ("D", "C", "Secondary: Species+OOD (D) vs OOD (C) [Species on OOD]"),
    ("B", "D", "Secondary: Species (B) vs Species+OOD (D) [Simplicity]"),
]

reconciliation_rows = []
for c1, c0, label in contrasts_spec:
    s1 = df_abl[df_abl["Config"] == c1].sort_values("Seed").reset_index(drop=True)
    s0 = df_abl[df_abl["Config"] == c0].sort_values("Seed").reset_index(drop=True)
    assert list(s1["Seed"]) == list(s0["Seed"]), f"Seed mismatch in {c1} vs {c0}"
    
    row = {
        "Comparison": label,
        "Config_1": c1,
        "Config_0": c0,
        "N_seeds": len(s1),
    }
    
    for m in METRICS:
        v1 = s1[m].values.astype(float)
        v0 = s0[m].values.astype(float)
        dv = v1 - v0
        delta = float(np.mean(dv))
        sd = float(np.std(dv, ddof=1))
        se = sd / np.sqrt(len(dv))
        
        t_val, p_val = stats.ttest_rel(v1, v0)
        w_val, p_w = stats.wilcoxon(dv, alternative="two-sided")
        dz = delta / sd if sd > 0 else 0.0
        
        ci_lo = delta - stats.t.ppf(0.975, df=len(dv)-1) * se
        ci_hi = delta + stats.t.ppf(0.975, df=len(dv)-1) * se
        
        row[f"{m}_delta"] = round(delta, 4)
        row[f"{m}_p_t"] = round(float(p_val), 4)
        row[f"{m}_p_w"] = round(float(p_w), 4)
        row[f"{m}_dz"] = round(dz, 4)
        row[f"{m}_ci95_lo"] = round(ci_lo, 4)
        row[f"{m}_ci95_hi"] = round(ci_hi, 4)
        
    reconciliation_rows.append(row)

df_reconcile = pd.DataFrame(reconciliation_rows)

# Multiple testing correction on Secondary QWK contrasts (excluding Primary D vs A)
sec_mask = df_reconcile["Comparison"].str.startswith("Secondary")
sec_qwk_ps = df_reconcile.loc[sec_mask, "QWK_p_t"].values
order_sec = np.argsort(sec_qwk_ps)
adj_sec_ps = np.zeros(len(sec_qwk_ps))
for rank, idx in enumerate(order_sec):
    adj_sec_ps[idx] = min(1.0, sec_qwk_ps[idx] * (len(sec_qwk_ps) - rank))
for i in range(len(order_sec) - 1, 0, -1):
    adj_sec_ps[order_sec[i-1]] = min(adj_sec_ps[order_sec[i-1]], adj_sec_ps[order_sec[i]])

df_reconcile["QWK_holm_adj_p"] = np.nan
df_reconcile.loc[sec_mask, "QWK_holm_adj_p"] = [round(p, 4) for p in adj_sec_ps]

df_reconcile.to_csv("outputs/casm_v2/phase45_ablation_reconciliation.csv", index=False)
print("Saved outputs/casm_v2/phase45_ablation_reconciliation.csv")

print("\n--- Summary of Key Contrasts ---")
for idx, r in df_reconcile.iterrows():
    p_adj_str = f", holm_adj_p={r['QWK_holm_adj_p']:.4f}" if not np.isnan(r['QWK_holm_adj_p']) else " (Primary)"
    print(f"{r['Comparison']}: Delta_QWK={r['QWK_delta']:+.4f}, p_t={r['QWK_p_t']:.4f}{p_adj_str}, d_z={r['QWK_dz']:+.4f}")
