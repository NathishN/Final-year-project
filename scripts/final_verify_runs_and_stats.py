"""
scripts/final_verify_runs_and_stats.py
======================================
Final (Phase 5.7 lock) independent verification of
  Part C : all 240 saved prediction tensors (4 configs x 30 seeds x 2 protocols), and
  Part D : the six paired contrasts (B-A, C-A, D-A under each protocol).

Independence from scripts/audit_gap1_metric_recomputation.py / audit_gap3_statistical_tests.py:
  * QWK is computed both with sklearn and with a from-scratch confusion-matrix formula.
  * Ordinal decoding is re-derived from the stored class probabilities
    (y_hat = sum_k [P(Y > k) > 0.5], P(Y > k) = sum_{j>k} p_j) and compared to stored predictions.
  * Exact Wilcoxon signed-rank p-values are obtained from an exact big-integer generating
    function over (doubled) mid-ranks -- a tie-aware conditional permutation distribution --
    and cross-checked against scipy (exact, no ties) and a 2,000,000-draw Monte-Carlo sign-flip test.
  * Holm-Bonferroni is implemented independently (step-down with monotonicity enforcement).
  * Split integrity (counts, duplicate IDs, session isolation) is computed from the manifests.

Statistical conventions: n = 30 paired seeds, ddof = 1 for SD, two-sided tests,
95% CI = mean +/- t_{0.975,29} * SD/sqrt(30), Cohen's d_z = mean/SD,
Wilcoxon two-sided p = min(1, 2 * P(W+ <= min(W+, W-))) under the conditional null.

Outputs: outputs/phase57_final_submission_lock/verification/runs_stats/
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "phase57_final_submission_lock", "verification", "runs_stats")
SEEDS = [42, 123, 777, 999, 2025, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47,
         53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
CONFIGS = ["A", "B", "C", "D"]
PROTOCOLS = {
    "historical": ("outputs/casm_v2/phase50_per_seed_predictions", "outputs/casm_v2/phase50_split_v2_manifest.csv"),
    "session_holdout": ("outputs/phase52_final_validation/session_holdout_retrain/predictions",
                        "outputs/phase52_final_validation/session_holdout_manifest.csv"),
}


def qwk_scratch(y, p, K=5):
    O = np.zeros((K, K))
    np.add.at(O, (y, p), 1)
    W = np.array([[(i - j) ** 2 for j in range(K)] for i in range(K)], float) / (K - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1.0 - (W * O).sum() / (W * E).sum()


def split_integrity(man):
    out = {}
    for s in ["train", "val", "test"]:
        out[f"n_{s}"] = int((man.split == s).sum())
    out["n_total"] = int(len(man))
    out["duplicate_image_paths"] = int(man.image_path.duplicated().sum())
    sess = {s: set(man.loc[man.split == s, "session_id"].astype(str)) for s in ["train", "val", "test"]}
    out["n_sessions_total"] = int(man.session_id.astype(str).nunique())
    out["sessions_train_test_overlap"] = sorted(sess["train"] & sess["test"])
    out["sessions_train_val_overlap"] = sorted(sess["train"] & sess["val"])
    out["sessions_val_test_overlap"] = sorted(sess["val"] & sess["test"])
    out["samples_in_train_from_test_sessions"] = int(man[(man.split == "train") &
                                                         man.session_id.astype(str).isin(sess["test"])].shape[0])
    out["samples_in_test_from_train_sessions"] = int(man[(man.split == "test") &
                                                         man.session_id.astype(str).isin(sess["train"])].shape[0])
    return out


# ---------------- exact Wilcoxon (independent implementation) ----------------
def wilcoxon_exact(diff):
    d = np.asarray(diff, float)
    n_zero = int((d == 0).sum())
    d = d[d != 0]
    a = np.abs(d)
    ranks = stats.rankdata(a, method="average")
    wp, wm = float(ranks[d > 0].sum()), float(ranks[d < 0].sum())
    r2 = [int(round(2 * r)) for r in ranks]  # doubled mid-ranks are integers
    # generating function prod_i (1 + x^{r_i}) with Python big integers (exact)
    poly = {0: 1}
    for r in r2:
        nxt = dict(poly)
        for k, v in poly.items():
            nxt[k + r] = nxt.get(k + r, 0) + v
        poly = nxt
    total = 2 ** len(r2)
    wmin2 = int(round(2 * min(wp, wm)))
    count_le = sum(v for k, v in poly.items() if k <= wmin2)
    p = min(1.0, 2 * count_le / total)
    _, cnt = np.unique(a, return_counts=True)
    return {"W_plus": wp, "W_minus": wm, "W": min(wp, wm), "n_zero": n_zero, "n_tied_groups": int((cnt > 1).sum()),
            "count_le": int(count_le), "total": int(total), "p_exact": p}


def mc_signflip(diff, n=2_000_000, seed=20261014):
    d = np.asarray(diff, float)
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    obs = min(r[d > 0].sum(), r[d < 0].sum())
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n:
        b = min(200_000, n - done)
        s = rng.integers(0, 2, size=(b, len(r)), dtype=np.int8)
        wp = s @ r
        wmin = np.minimum(wp, r.sum() - wp)
        hits += int((wmin <= obs + 1e-9).sum())
        done += b
    return hits / n


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[idx]))
        adj[idx] = run
    return adj


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, integrity = [], {}
    mans = {}
    for proto, (pdir, mpath) in PROTOCOLS.items():
        man = pd.read_csv(os.path.join(ROOT, mpath), low_memory=False)
        mans[proto] = man
        integrity[proto] = split_integrity(man)
        test = man[man.split == "test"]
        y_man = test.severity_grade.values.astype(int)
        for cfg in CONFIGS:
            for s in SEEDS:
                f = os.path.join(ROOT, pdir, f"preds_{cfg}_seed{s}.pt")
                d = torch.load(f, map_location="cpu", weights_only=False)
                p = d["predictions"].numpy().astype(int)
                y = d["targets"].numpy().astype(int)
                pr = d["probabilities"].numpy().astype(np.float64)
                cum = np.stack([pr[:, k + 1:].sum(1) for k in range(4)], 1)
                dec = (cum > 0.5).sum(1)
                q_sk = cohen_kappa_score(y, p, weights="quadratic")
                q_sc = qwk_scratch(y, p)
                mae = np.abs(y - p).mean()
                acc = (y == p).mean() * 100
                f1 = f1_score(y, p, average="macro", labels=list(range(5)), zero_division=0) * 100
                checks = {
                    "n_matches_manifest": len(p) == len(y_man),
                    "targets_match_manifest_order": bool(np.array_equal(y, y_man)),
                    "config_seed_match_filename": (d.get("config") == cfg and int(d.get("seed")) == s),
                    "pred_range_ok": bool(p.min() >= 0 and p.max() <= 4),
                    "prob_shape_ok": pr.shape == (len(p), 5),
                    "probs_sum_to_1": bool(np.allclose(pr.sum(1), 1, atol=1e-4)),
                }
                dec_agree = float((dec == p).mean())
                stored = {"qwk": float(d.get("qwk", np.nan)), "mae": float(d.get("mae", np.nan)),
                          "acc": float(d.get("accuracy", np.nan))}
                diffs = {"qwk": abs(stored["qwk"] - q_sk), "mae": abs(stored["mae"] - mae), "acc": abs(stored["acc"] - acc)}
                ok = (all(checks.values()) and abs(q_sk - q_sc) < 1e-10 and diffs["qwk"] <= 5.0001e-5
                      and diffs["mae"] <= 5.0001e-5 and diffs["acc"] <= 5.0001e-3)
                rows.append({"protocol": proto, "config": cfg, "seed": s, "n_test": len(p),
                             "stored_qwk_4dp": stored["qwk"], "recomputed_qwk": q_sk, "qwk_scratch": q_sc,
                             "abs_diff_qwk": diffs["qwk"], "stored_mae_4dp": stored["mae"], "recomputed_mae": mae,
                             "abs_diff_mae": diffs["mae"], "stored_acc_2dp": stored["acc"], "recomputed_acc_pct": acc,
                             "abs_diff_acc": diffs["acc"], "recomputed_macro_f1_pct": f1,
                             "decode_from_probs_agreement": dec_agree, **checks,
                             "status": "PASS" if ok else "FAIL"})
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(OUT, "per_run_metric_verification.csv"), index=False)

    # cross-protocol comparison of test sets
    th, ts = mans["historical"], mans["session_holdout"]
    assert np.array_equal(th.image_path.values, ts.image_path.values), "manifests not row-aligned"
    moved = th[(th.split != ts.split)]
    integrity["cross_protocol"] = {
        "rows_with_different_split": int(len(moved)),
        "moved_from_to": {f"{a}->{b}": int(c) for (a, b), c in
                          pd.Series(list(zip(th.split[th.split != ts.split], ts.split[th.split != ts.split]))).value_counts().items()},
        "moved_sessions": sorted(moved.session_id.astype(str).unique().tolist()),
        "test_set_identity_hist_subset_of_holdout": bool(set(th.image_path[th.split == "test"]) <=
                                                         set(ts.image_path[ts.split == "test"])),
    }
    with open(os.path.join(OUT, "split_integrity.json"), "w") as fh:
        json.dump(integrity, fh, indent=2)
    pd.DataFrame([{"protocol": k, **{kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}}
                  for k, v in integrity.items() if k != "cross_protocol"]).to_csv(
        os.path.join(OUT, "split_integrity_summary.csv"), index=False)

    # aggregates
    agg = []
    for proto in PROTOCOLS:
        for cfg in CONFIGS:
            s = runs[(runs.protocol == proto) & (runs.config == cfg)]
            assert len(s) == 30 and s.seed.is_unique
            agg.append({"protocol": proto, "config": cfg, "n_seeds": len(s), "n_test": int(s.n_test.iloc[0]),
                        "qwk_mean": s.recomputed_qwk.mean(), "qwk_sd_ddof1": s.recomputed_qwk.std(ddof=1),
                        "qwk_min": s.recomputed_qwk.min(), "qwk_max": s.recomputed_qwk.max(),
                        "mae_mean": s.recomputed_mae.mean(), "mae_sd_ddof1": s.recomputed_mae.std(ddof=1),
                        "acc_mean_pct": s.recomputed_acc_pct.mean(), "acc_sd_ddof1": s.recomputed_acc_pct.std(ddof=1),
                        "macro_f1_mean_pct": s.recomputed_macro_f1_pct.mean(),
                        "macro_f1_sd_ddof1": s.recomputed_macro_f1_pct.std(ddof=1)})
    aggdf = pd.DataFrame(agg)
    aggdf.to_csv(os.path.join(OUT, "aggregate_metrics_verified.csv"), index=False)

    # ---------------- Part D ----------------
    con = []
    tcrit = stats.t.ppf(0.975, 29)
    for proto in PROTOCOLS:
        piv = runs[runs.protocol == proto].pivot(index="seed", columns="config", values="recomputed_qwk").loc[SEEDS]
        block = []
        for c in ["B", "C", "D"]:
            diff = (piv[c] - piv["A"]).values
            n = len(diff)
            m, sd = diff.mean(), diff.std(ddof=1)
            se = sd / np.sqrt(n)
            t = m / se
            pt = 2 * stats.t.sf(abs(t), n - 1)
            w = wilcoxon_exact(diff)
            sp = stats.wilcoxon(diff, method="exact").pvalue if w["n_tied_groups"] == 0 and w["n_zero"] == 0 else np.nan
            block.append({"protocol": proto, "contrast": f"{c}-A", "n_pairs": n, "mean_diff": m, "sd_diff": sd,
                          "ci95_low": m - tcrit * se, "ci95_high": m + tcrit * se, "cohen_dz": m / sd,
                          "t_stat": t, "df": n - 1, "p_t_raw": pt,
                          "n_pos": int((diff > 0).sum()), "n_neg": int((diff < 0).sum()), "n_zero": w["n_zero"],
                          "n_tied_abs_groups": w["n_tied_groups"], "W_plus": w["W_plus"], "W_minus": w["W_minus"],
                          "W": w["W"], "exact_count_le": w["count_le"], "exact_total": w["total"],
                          "p_wilcoxon_exact": w["p_exact"], "p_wilcoxon_scipy_exact": sp,
                          "p_wilcoxon_mc_signflip_2e6": mc_signflip(diff),
                          "p_wilcoxon_asymptotic": stats.wilcoxon(diff, method="approx", correction=False).pvalue})
        ht = holm([b["p_t_raw"] for b in block])
        hw = holm([b["p_wilcoxon_exact"] for b in block])
        for b, a1, a2 in zip(block, ht, hw):
            b["p_t_holm"] = a1
            b["p_wilcoxon_exact_holm"] = a2
            b["sig_holm_wilcoxon_0.05"] = bool(a2 < 0.05)
            b["sig_holm_wilcoxon_0.001"] = bool(a2 < 0.001)
            b["sig_holm_t_0.05"] = bool(a1 < 0.05)
            b["sig_holm_t_0.001"] = bool(a1 < 0.001)
            con.append(b)
    cdf = pd.DataFrame(con)
    cdf.to_csv(os.path.join(OUT, "paired_contrasts_verified.csv"), index=False)

    # Rounded-metadata sensitivity for historical B-A (4-dp stored QWK)
    h = runs[runs.protocol == "historical"].pivot(index="seed", columns="config", values="stored_qwk_4dp").loc[SEEDS]
    wr = wilcoxon_exact((h["B"] - h["A"]).values)
    sens = {"historical_B-A_from_4dp_stored_qwk": wr}

    # cross-check against the existing Gap-3 output
    g3p = os.path.join(ROOT, "outputs", "final_submission", "tables", "gap3_statistical_contrasts_verified.csv")
    if os.path.exists(g3p):
        g3 = pd.read_csv(g3p)
        mx = 0.0
        for i, r in cdf.iterrows():
            gr = g3.iloc[i]
            mx = max(mx, abs(gr.p_exact_conditional - r.p_wilcoxon_exact) / r.p_wilcoxon_exact,
                     abs(gr.mean_diff - r.mean_diff), abs(gr.t_stat - r.t_stat))
        sens["max_discrepancy_vs_existing_gap3_script"] = mx

    summary = {"n_runs": int(len(runs)), "n_pass": int((runs.status == "PASS").sum()),
               "max_abs_diff_qwk": float(runs.abs_diff_qwk.max()), "max_abs_diff_mae": float(runs.abs_diff_mae.max()),
               "max_abs_diff_acc": float(runs.abs_diff_acc.max()),
               "max_qwk_sklearn_vs_scratch": float((runs.recomputed_qwk - runs.qwk_scratch).abs().max()),
               "min_decode_agreement": float(runs.decode_from_probs_agreement.min()),
               "sensitivity": sens}
    with open(os.path.join(OUT, "runs_stats_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    pd.set_option("display.width", 250)
    print(json.dumps(summary, indent=2, default=float))
    print(json.dumps(integrity, indent=2)[:3000])
    print(aggdf.round(4).to_string(index=False))
    print(cdf[["protocol", "contrast", "mean_diff", "ci95_low", "ci95_high", "cohen_dz", "t_stat", "p_t_raw", "p_t_holm",
               "n_pos", "n_neg", "W_plus", "W_minus", "p_wilcoxon_exact", "p_wilcoxon_scipy_exact",
               "p_wilcoxon_mc_signflip_2e6", "p_wilcoxon_exact_holm"]].to_string(index=False))
    if (runs.status != "PASS").any():
        print(runs[runs.status != "PASS"].to_string())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
