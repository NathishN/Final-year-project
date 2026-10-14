import os
import torch
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

HIST_PRED_DIR = "outputs/casm_v2/phase50_per_seed_predictions"
RETRAIN_CSV = "outputs/phase52_final_validation/session_holdout_retrain/per_seed_results.csv"

SEEDS = [42, 123, 777, 999, 2025, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
CONFIGS = ["A", "B", "C", "D"]

# 1. Historical Split V2
hist_qwks = {cfg: {} for cfg in CONFIGS}
for cfg in CONFIGS:
    for s in SEEDS:
        fpath = os.path.join(HIST_PRED_DIR, f"preds_{cfg}_seed{s}.pt")
        pt = torch.load(fpath, map_location="cpu", weights_only=True)
        hist_qwks[cfg][s] = float(pt["qwk"])

arrA = np.array([hist_qwks["A"][s] for s in SEEDS])
arrB = np.array([hist_qwks["B"][s] for s in SEEDS])
diff_BA = arrB - arrA

print("=== HISTORICAL B - A ===")
for s, a, b, d in zip(SEEDS, arrA, arrB, diff_BA):
    print(f"Seed {s:4d}: A={a:.6f}, B={b:.6f}, B-A={d:+.6f}")

print("\nBasic Stats B-A:")
print(f"Mean diff: {np.mean(diff_BA):.6f}")
print(f"Std diff (ddof=1): {np.std(diff_BA, ddof=1):.6f}")
t_res = scipy_stats.ttest_1samp(diff_BA, 0.0)
print(f"t-test: t={t_res.statistic:.6f}, p={t_res.pvalue:.8e}")

abs_d = np.abs(diff_BA)
ranks = scipy_stats.rankdata(abs_d)
print("\nRanks of |B - A|:")
for s, d, ad, r in zip(SEEDS, diff_BA, abs_d, ranks):
    print(f"Seed {s:4d}: diff={d:+.6f}, |diff|={ad:.6f}, rank={r}")

pos_ranks = ranks[diff_BA > 0]
neg_ranks = ranks[diff_BA < 0]
print(f"\nPositive rank sum: {np.sum(pos_ranks)} (N={len(pos_ranks)})")
print(f"Negative rank sum: {np.sum(neg_ranks)} (N={len(neg_ranks)})")
print(f"Total rank sum: {np.sum(ranks)} (Expected 30*31/2 = 465)")

# Check ties in abs_d
unique, counts = np.unique(abs_d, return_counts=True)
ties = unique[counts > 1]
print(f"\nTied values in |diff|: {ties}")
for t in ties:
    print(f"Tied val: {t}, count: {counts[unique == t][0]}")

print("\n--- Scipy wilcoxon variations for Historical B - A ---")
for zm in ["wilcox", "pratt", "zsplit"]:
    for cor in [False, True]:
        for alt in ["two-sided", "less", "greater"]:
            w = scipy_stats.wilcoxon(diff_BA, zero_method=zm, correction=cor, alternative=alt)
            print(f"zm={zm:6s}, cor={str(cor):5s}, alt={alt:9s} -> stat={w.statistic}, p={w.pvalue:.8e}")

# What if using exact method?
try:
    w_exact = scipy_stats.wilcoxon(diff_BA, method="exact")
    print(f"method='exact' -> stat={w_exact.statistic}, p={w_exact.pvalue:.8e}")
except Exception as e:
    print(f"exact failed: {e}")

try:
    w_approx = scipy_stats.wilcoxon(diff_BA, method="approx")
    print(f"method='approx' -> stat={w_approx.statistic}, p={w_approx.pvalue:.8e}")
except Exception as e:
    print(f"approx failed: {e}")

# Where could W=72.0 and p=0.000555 come from?
# Could rounded differences or an earlier version of scipy produce 72.0 and 0.000555?
diff_BA_round4 = np.round(diff_BA, 4)
diff_BA_round5 = np.round(diff_BA, 5)
print("\nTesting rounding:")
w_r4 = scipy_stats.wilcoxon(diff_BA_round4)
print(f"Round 4: stat={w_r4.statistic}, p={w_r4.pvalue:.8e}")
w_r5 = scipy_stats.wilcoxon(diff_BA_round5)
print(f"Round 5: stat={w_r5.statistic}, p={w_r5.pvalue:.8e}")

# Test if ranks were integer (no tie averaging)
# If tied pair got integer ranks 13 and 14:
print("\nIf tied pair ranks were 13, 14 instead of 13.5, 13.5:")
# Suppose one of the tied was positive?
# Let's see which seed had the tie!
