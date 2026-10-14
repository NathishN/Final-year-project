"""
scripts/audit_gap3_statistical_tests.py
Independently verifies all paired statistical hypothesis tests and exact conditional Wilcoxon signed-rank p-values
using the verified per-seed QWK metrics from Gap 1.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

SEEDS = [42, 123, 777, 999, 2025, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
CONFIGS = ["A", "B", "C", "D"]
CONTRASTS = ["B", "C", "D"]

def exact_conditional_wilcoxon(diff):
    n = len(diff)
    nonzero_mask = (diff != 0)
    d = diff[nonzero_mask]
    n_nonzero = len(d)
    
    abs_d = np.abs(d)
    ranks = scipy_stats.rankdata(abs_d, method="average")
    
    w_pos = float(np.sum(ranks[d > 0]))
    w_neg = float(np.sum(ranks[d < 0]))
    w_stat = float(min(w_pos, w_neg))
    
    # Scale ranks by 2 to ensure all ranks are exact integers
    ranks_int = np.round(2 * ranks).astype(int)
    total_int = int(np.sum(ranks_int))
    w_pos_int = int(np.round(2 * w_pos))
    w_neg_int = int(np.round(2 * w_neg))
    w_stat_int = min(w_pos_int, w_neg_int)
    
    # Dynamic programming for generating subset sum distribution of ranks
    dp = np.zeros(total_int + 1, dtype=np.int64)
    dp[0] = 1
    for r in ranks_int:
        new_dp = np.zeros_like(dp)
        new_dp += dp
        new_dp[r:] += dp[:-r if r > 0 else len(dp)]
        dp = new_dp
        
    total_comb = 2**n_nonzero
    count_le = int(np.sum(dp[:w_stat_int + 1]))
    count_ge = int(np.sum(dp[total_int - w_stat_int:]))
    count_two_sided = count_le + count_ge
    p_exact_conditional = float(count_two_sided / total_comb)
    
    # SciPy comparisons
    scipy_exact = float(scipy_stats.wilcoxon(diff, method="exact").pvalue)
    scipy_approx = float(scipy_stats.wilcoxon(diff, method="approx", correction=False).pvalue)
    
    # Unique ties check
    unique_vals, val_counts = np.unique(abs_d, return_counts=True)
    tied_vals = unique_vals[val_counts > 1]
    
    return {
        "n_nonzero": n_nonzero,
        "n_zeros": n - n_nonzero,
        "ranks": ranks,
        "w_pos": w_pos,
        "w_neg": w_neg,
        "w_stat": w_stat,
        "total_combinations": total_comb,
        "favorable_combinations": count_two_sided,
        "count_le": count_le,
        "count_ge": count_ge,
        "p_exact_conditional": p_exact_conditional,
        "scipy_exact_noties": scipy_exact,
        "scipy_approx": scipy_approx,
        "tied_values": [float(t) for t in tied_vals],
        "n_positive": int(np.sum(diff > 0)),
        "n_negative": int(np.sum(diff < 0))
    }

def main():
    print("Executing Gap 3: Paired Statistical Hypothesis Tests & Exact Conditional Wilcoxon Verification...")
    
    in_csv = "outputs/final_submission/tables/gap1_per_seed_metric_verification.csv"
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Missing input {in_csv}")
    df = pd.read_csv(in_csv)
    
    results = []
    
    for proto in ["Historical Split V2", "100% Group Session Holdout"]:
        proto_df = df[df["protocol"] == proto]
        piv = proto_df.pivot(index="seed", columns="config", values="recomputed_qwk").loc[SEEDS]
        
        raw_p_t = []
        raw_p_w = []
        contrast_data = []
        
        for c in CONTRASTS:
            qwk_a = piv["A"].values
            qwk_c = piv[c].values
            diff = qwk_c - qwk_a
            
            n = len(diff)
            mean_d = float(np.mean(diff))
            std_d = float(np.std(diff, ddof=1))
            se_d = std_d / np.sqrt(n)
            t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
            ci_95 = [float(mean_d - t_crit * se_d), float(mean_d + t_crit * se_d)]
            cohen_dz = float(mean_d / std_d)
            
            t_res = scipy_stats.ttest_1samp(diff, 0.0)
            t_stat = float(t_res.statistic)
            p_t = float(t_res.pvalue)
            
            w_res = exact_conditional_wilcoxon(diff)
            
            raw_p_t.append(p_t)
            raw_p_w.append(w_res["p_exact_conditional"])
            
            contrast_data.append({
                "protocol": proto,
                "contrast": f"{c} vs A",
                "config": c,
                "n_seeds": n,
                "df": n - 1,
                "mean_diff": mean_d,
                "std_diff": std_d,
                "se_diff": se_d,
                "ci_95_low": ci_95[0],
                "ci_95_high": ci_95[1],
                "cohen_dz": cohen_dz,
                "t_stat": t_stat,
                "p_t_raw": p_t,
                "wilcoxon_stat_W": w_res["w_stat"],
                "w_pos": w_res["w_pos"],
                "w_neg": w_res["w_neg"],
                "p_exact_conditional": w_res["p_exact_conditional"],
                "p_scipy_exact_noties": w_res["scipy_exact_noties"],
                "p_asymptotic": w_res["scipy_approx"],
                "tied_values": w_res["tied_values"],
                "favorable_combinations": w_res["favorable_combinations"],
                "total_combinations": w_res["total_combinations"],
                "pairs_pos_neg": f"{w_res['n_positive']} / {w_res['n_negative']}"
            })
            
        # Holm-Bonferroni correction within protocol (family of m=3 contrasts)
        m = len(CONTRASTS)
        # t-test Holm
        sort_t_idx = np.argsort(raw_p_t)
        holm_t = np.zeros(m)
        for r_i, o_i in enumerate(sort_t_idx):
            holm_t[o_i] = min(1.0, raw_p_t[o_i] * (m - r_i))
        for i in range(1, m):
            curr_i = sort_t_idx[i]
            prev_i = sort_t_idx[i-1]
            holm_t[curr_i] = max(holm_t[curr_i], holm_t[prev_i])
            
        # Wilcoxon exact Holm
        sort_w_idx = np.argsort(raw_p_w)
        holm_w = np.zeros(m)
        for r_i, o_i in enumerate(sort_w_idx):
            holm_w[o_i] = min(1.0, raw_p_w[o_i] * (m - r_i))
        for i in range(1, m):
            curr_i = sort_w_idx[i]
            prev_i = sort_w_idx[i-1]
            holm_w[curr_i] = max(holm_w[curr_i], holm_w[prev_i])
            
        for i, cd in enumerate(contrast_data):
            cd["p_t_holm"] = float(holm_t[i])
            cd["p_wilcoxon_exact_holm"] = float(holm_w[i])
            cd["significant_alpha_05"] = bool(holm_w[i] < 0.05)
            cd["significant_alpha_001"] = bool(holm_w[i] < 0.001)
            results.append(cd)
            
    res_df = pd.DataFrame(results)
    out_csv = "outputs/final_submission/tables/gap3_statistical_contrasts_verified.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Saved verified statistical contrasts to: {out_csv}")
    
    print("\n=== VERIFIED SIX-CONTRAST STATISTICAL TABLE ===")
    for _, r in res_df.iterrows():
        print(f"[{r['protocol']}] Contrast {r['contrast']}:")
        print(f"  Delta QWK: {r['mean_diff']:.4f} [95% CI: {r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}], Cohen's dz: {r['cohen_dz']:.4f}")
        print(f"  Paired t(29): {r['t_stat']:.4f}, raw p: {r['p_t_raw']:.6e}, Holm p: {r['p_t_holm']:.6e}")
        print(f"  Wilcoxon W: {r['wilcoxon_stat_W']:.1f} (W+={r['w_pos']:.1f}, W-={r['w_neg']:.1f})")
        print(f"  Exact conditional permutation p: {r['p_exact_conditional']:.8e} (Holm p: {r['p_wilcoxon_exact_holm']:.8e})")
        print(f"  SciPy exact (no-ties lookup):   {r['p_scipy_exact_noties']:.8e}")
        print(f"  Asymptotic normal p:            {r['p_asymptotic']:.8e}")
        print(f"  Tied |diff| values:             {r['tied_values']}")
        print(f"  Favorable sign combinations:    {r['favorable_combinations']} / {r['total_combinations']}")
        print(f"  Significant at alpha=0.05:      {r['significant_alpha_05']}")
        print(f"  Significant at alpha=0.001:     {r['significant_alpha_001']}\n")

if __name__ == "__main__":
    main()
