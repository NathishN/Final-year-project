"""
compute_exact_permutation_tests.py
Computes mathematically exact conditional permutation tests under observed ranks (with ties)
for all contrasts across both Historical Split V2 and 100% Group Session Holdout.
"""

import os
import torch
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

HIST_PRED_DIR = "outputs/casm_v2/phase50_per_seed_predictions"
RETRAIN_CSV = "outputs/phase52_final_validation/session_holdout_retrain/per_seed_results.csv"
SEEDS = [42, 123, 777, 999, 2025, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
CONFIGS = ["A", "B", "C", "D"]

def exact_permutation_test(diff):
    abs_d = np.abs(diff)
    ranks = scipy_stats.rankdata(abs_d, method="average")
    w_pos = np.sum(ranks[diff > 0])
    w_neg = np.sum(ranks[diff < 0])
    
    # Scale ranks by 2 to make all integers (since ties can end in .5)
    ranks_int = np.round(2 * ranks).astype(int)
    total_int = int(np.sum(ranks_int))
    w_pos_int = int(np.round(2 * w_pos))
    w_neg_int = int(np.round(2 * w_neg))
    w_stat_int = min(w_pos_int, w_neg_int)
    
    dp = np.zeros(total_int + 1, dtype=np.int64)
    dp[0] = 1
    for r in ranks_int:
        new_dp = np.zeros_like(dp)
        new_dp += dp
        new_dp[r:] += dp[:-r if r > 0 else len(dp)]
        dp = new_dp
        
    total_comb = 2**len(diff)
    count_le = int(np.sum(dp[:w_stat_int + 1]))
    count_ge = int(np.sum(dp[total_int - w_stat_int:]))
    count_two_sided = count_le + count_ge
    p_exact = count_two_sided / total_comb
    
    scipy_exact = float(scipy_stats.wilcoxon(diff, method="exact").pvalue)
    scipy_approx = float(scipy_stats.wilcoxon(diff, method="approx", correction=False).pvalue)
    
    # Detect ties
    unique, counts = np.unique(abs_d, return_counts=True)
    ties = unique[counts > 1]
    
    return {
        "w_pos": float(w_pos),
        "w_neg": float(w_neg),
        "w_stat": float(min(w_pos, w_neg)),
        "count_le": count_le,
        "count_ge": count_ge,
        "count_two_sided": count_two_sided,
        "p_exact_perm": float(p_exact),
        "scipy_exact": scipy_exact,
        "scipy_approx": scipy_approx,
        "ties": [float(t) for t in ties]
    }

def main():
    print("=== EXACT CONDITIONAL PERMUTATION TEST ANALYSIS ===\n")
    
    # 1. Historical Split V2
    print("--- Historical Split V2 (N_test=12,833, n_seeds=30) ---")
    hist_qwks = {cfg: [float(torch.load(f"{HIST_PRED_DIR}/preds_{cfg}_seed{s}.pt", map_location="cpu", weights_only=True)["qwk"]) for s in SEEDS] for cfg in CONFIGS}
    
    for c in ["B", "C", "D"]:
        diff = np.array(hist_qwks[c]) - np.array(hist_qwks["A"])
        res = exact_permutation_test(diff)
        print(f"\nContrast {c} vs A:")
        print(f"  W+ (pos rank sum): {res['w_pos']:.1f}")
        print(f"  W- (neg rank sum): {res['w_neg']:.1f}")
        print(f"  W (min rank sum):  {res['w_stat']:.1f}")
        print(f"  Tied values in |diff|: {res['ties']}")
        print(f"  Permutations <= W: {res['count_le']} / {2**30}")
        print(f"  Exact Conditional Permutation p-value: {res['p_exact_perm']:.8e} ({res['p_exact_perm']:.6f})")
        print(f"  SciPy method='exact' (no-ties lookup): {res['scipy_exact']:.8e} ({res['scipy_exact']:.6f})")
        print(f"  SciPy method='approx' (asymptotic):     {res['scipy_approx']:.8e} ({res['scipy_approx']:.6f})")

    # 2. 100% Group Session Holdout
    print("\n--- 100% Group Session Holdout (N_test=12,980, n_seeds=30) ---")
    df_retrain = pd.read_csv(RETRAIN_CSV)
    piv_retrain = df_retrain.pivot(index="seed", columns="config", values="test_qwk")
    retrain_qwks = {cfg: [float(piv_retrain.loc[s, cfg]) for s in SEEDS] for cfg in CONFIGS}
    
    for c in ["B", "C", "D"]:
        diff = np.array(retrain_qwks[c]) - np.array(retrain_qwks["A"])
        res = exact_permutation_test(diff)
        print(f"\nContrast {c} vs A:")
        print(f"  W+ (pos rank sum): {res['w_pos']:.1f}")
        print(f"  W- (neg rank sum): {res['w_neg']:.1f}")
        print(f"  W (min rank sum):  {res['w_stat']:.1f}")
        print(f"  Tied values in |diff|: {res['ties']}")
        print(f"  Permutations <= W: {res['count_le']} / {2**30}")
        print(f"  Exact Conditional Permutation p-value: {res['p_exact_perm']:.8e} ({res['p_exact_perm']:.6f})")
        print(f"  SciPy method='exact' (no-ties lookup): {res['scipy_exact']:.8e} ({res['scipy_exact']:.6f})")
        print(f"  SciPy method='approx' (asymptotic):     {res['scipy_approx']:.8e} ({res['scipy_approx']:.6f})")

if __name__ == "__main__":
    main()
