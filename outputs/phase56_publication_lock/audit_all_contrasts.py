import os
import torch
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

HIST_PRED_DIR = "outputs/casm_v2/phase50_per_seed_predictions"
RETRAIN_CSV = "outputs/phase52_final_validation/session_holdout_retrain/per_seed_results.csv"

SEEDS = [42, 123, 777, 999, 2025, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
CONFIGS = ["A", "B", "C", "D"]

def audit_dataset(name, qwk_dict):
    print(f"\n==================== {name} ====================")
    arrA = np.array([qwk_dict["A"][s] for s in SEEDS])
    for c in ["B", "C", "D"]:
        arr = np.array([qwk_dict[c][s] for s in SEEDS])
        diff = arr - arrA
        n = len(diff)
        mean_d = np.mean(diff)
        std_d = np.std(diff, ddof=1)
        se_d = std_d / np.sqrt(n)
        t_res = scipy_stats.ttest_1samp(diff, 0.0)
        
        abs_d = np.abs(diff)
        ranks = scipy_stats.rankdata(abs_d)
        pos_ranks = ranks[diff > 0]
        neg_ranks = ranks[diff < 0]
        pos_sum = np.sum(pos_ranks)
        neg_sum = np.sum(neg_ranks)
        w_stat_scipy = scipy_stats.wilcoxon(diff).statistic
        
        # Ties
        unique, counts = np.unique(abs_d, return_counts=True)
        ties = unique[counts > 1]
        
        # Exact vs Approx
        try:
            w_exact = scipy_stats.wilcoxon(diff, method="exact")
            p_exact = w_exact.pvalue
        except Exception as e:
            p_exact = None
            
        w_approx = scipy_stats.wilcoxon(diff, method="approx", correction=False)
        p_approx_nocor = w_approx.pvalue
        
        w_approx_cor = scipy_stats.wilcoxon(diff, method="approx", correction=True)
        p_approx_cor = w_approx_cor.pvalue
        
        print(f"\n--- Contrast {c} vs A ---")
        print(f"Mean diff: {mean_d:+.6f}, Std diff: {std_d:.6f}, 95% CI: [{mean_d - 2.04523*se_d:.6f}, {mean_d + 2.04523*se_d:.6f}]")
        print(f"t-stat: {t_res.statistic:.6f}, df: {n-1}, p_t: {t_res.pvalue:.8e}")
        print(f"Ranks: Pos sum={pos_sum}, Neg sum={neg_sum}, Scipy stat={w_stat_scipy}")
        print(f"Ties count: {len(ties)} (Tied values: {ties})")
        print(f"Wilcoxon exact p-value: {p_exact:.8e}" if p_exact is not None else "Wilcoxon exact p: N/A")
        print(f"Wilcoxon approx (no continuity correction) p-value: {p_approx_nocor:.8e}")
        print(f"Wilcoxon approx (continuity correction) p-value:    {p_approx_cor:.8e}")
        print(f"Number of positive diffs: {np.sum(diff > 0)}, negative: {np.sum(diff < 0)}, zero: {np.sum(diff == 0)}")

# Historical
hist_qwks = {cfg: {} for cfg in CONFIGS}
for cfg in CONFIGS:
    for s in SEEDS:
        fpath = os.path.join(HIST_PRED_DIR, f"preds_{cfg}_seed{s}.pt")
        pt = torch.load(fpath, map_location="cpu", weights_only=True)
        hist_qwks[cfg][s] = float(pt["qwk"])

audit_dataset("Historical Split V2 (N=12,833)", hist_qwks)

# Session Held-Out
df_retrain = pd.read_csv(RETRAIN_CSV)
piv_retrain = df_retrain.pivot(index="seed", columns="config", values="test_qwk")
retrain_qwks = {cfg: {s: float(piv_retrain.loc[s, cfg]) for s in SEEDS} for cfg in CONFIGS}

audit_dataset("100% Group Session Holdout (N=12,980)", retrain_qwks)
