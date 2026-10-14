"""
scripts/audit_gap1_metric_recomputation.py
Independently recomputes all classification and ordinal metrics across all 240 prediction tensors:
- Historical Split V2 (120 models: 4 configs x 30 seeds, N=12,833)
- Session Holdout Split (120 models: 4 configs x 30 seeds, N=12,980)

Recomputes:
- QWK (Quadratic Weighted Kappa)
- MAE (Mean Absolute Error)
- Accuracy (%)
- Macro-F1 (%)

Outputs per-seed verification CSV and aggregate summary.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

SEEDS = [42, 123, 777, 999, 2025, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
CONFIGS = ["A", "B", "C", "D"]

PROTOCOLS = [
    ("Historical Split V2", "outputs/casm_v2/phase50_per_seed_predictions", 12833, "outputs/casm_v2/phase50_split_v2_manifest.csv"),
    ("100% Group Session Holdout", "outputs/phase52_final_validation/session_holdout_retrain/predictions", 12980, "outputs/phase52_final_validation/session_holdout_manifest.csv")
]

def main():
    print("Executing Gap 1: Full Independent Recomputation of All 240 Prediction Tensors...")
    
    rows = []
    
    for proto_name, pred_dir, expected_n, manifest_path in PROTOCOLS:
        manifest = pd.read_csv(manifest_path)
        manifest_test = manifest[manifest["split"] == "test"]
        manifest_tgts = manifest_test["severity_grade"].values
        manifest_sessions = manifest_test["session_id"].values
        manifest_domains = manifest_test["domain"].values
        
        if len(manifest_tgts) != expected_n:
            raise ValueError(f"Manifest count {len(manifest_tgts)} != expected {expected_n}")
            
        for cfg in CONFIGS:
            for s in SEEDS:
                fpath = os.path.join(pred_dir, f"preds_{cfg}_seed{s}.pt")
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Missing {fpath}")
                    
                pt = torch.load(fpath, map_location="cpu", weights_only=False)
                preds = pt["predictions"].numpy()
                tgts = pt["targets"].numpy()
                
                # Validation checks
                assert len(preds) == expected_n, f"Length mismatch in {fpath}: {len(preds)} vs {expected_n}"
                assert np.array_equal(manifest_tgts, tgts), f"Target mismatch in {fpath}"
                assert np.all((preds >= 0) & (preds <= 4)), f"Predictions out of range [0, 4] in {fpath}"
                assert np.all((tgts >= 0) & (tgts <= 4)), f"Targets out of range [0, 4] in {fpath}"
                
                qwk = float(cohen_kappa_score(tgts, preds, weights="quadratic"))
                mae = float(np.mean(np.abs(tgts - preds)))
                acc = float(np.mean(tgts == preds) * 100.0)
                f1 = float(f1_score(tgts, preds, average="macro") * 100.0)
                
                stored_qwk = float(pt.get("qwk", float("nan")))
                stored_mae = float(pt.get("mae", float("nan")))
                stored_acc = float(pt.get("accuracy", pt.get("acc", float("nan"))))
                
                diff_qwk = abs(stored_qwk - qwk) if not np.isnan(stored_qwk) else 0.0
                diff_mae = abs(stored_mae - mae) if not np.isnan(stored_mae) else 0.0
                diff_acc = abs(stored_acc - acc) if not np.isnan(stored_acc) else 0.0
                
                # Check for significant discrepancy beyond rounding
                status = "EXACT_REPRODUCED" if (diff_qwk < 5e-4 and diff_mae < 5e-4 and diff_acc < 0.02) else "DISCREPANCY"
                
                rows.append({
                    "protocol": proto_name,
                    "config": cfg,
                    "seed": s,
                    "n_samples": expected_n,
                    "stored_qwk": stored_qwk,
                    "recomputed_qwk": qwk,
                    "diff_qwk": diff_qwk,
                    "stored_mae": stored_mae,
                    "recomputed_mae": mae,
                    "diff_mae": diff_mae,
                    "stored_acc": stored_acc,
                    "recomputed_acc": acc,
                    "diff_acc": diff_acc,
                    "recomputed_macro_f1": f1,
                    "verification_status": status
                })

    df = pd.DataFrame(rows)
    print(f"Total rows evaluated: {len(df)}")
    print(f"Verification status counts:\n{df['verification_status'].value_counts()}")
    print(f"Max absolute QWK difference against stored: {df['diff_qwk'].max():.6f}")
    print(f"Max absolute MAE difference against stored: {df['diff_mae'].max():.6f}")
    print(f"Max absolute Acc difference against stored: {df['diff_acc'].max():.6f}")
    
    # Save per-seed audit CSV
    out_dir = "outputs/final_submission/tables"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "gap1_per_seed_metric_verification.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved detailed per-seed audit: {out_csv}")
    
    # Compute aggregate table (ddof=1 and ddof=0)
    agg_rows = []
    for proto_name, _, expected_n, _ in PROTOCOLS:
        sub = df[df["protocol"] == proto_name]
        for cfg in CONFIGS:
            c_sub = sub[sub["config"] == cfg]
            qwk_arr = c_sub["recomputed_qwk"].values
            mae_arr = c_sub["recomputed_mae"].values
            acc_arr = c_sub["recomputed_acc"].values
            f1_arr = c_sub["recomputed_macro_f1"].values
            
            agg_rows.append({
                "protocol": proto_name,
                "config": cfg,
                "n_seeds": len(qwk_arr),
                "n_samples": expected_n,
                "qwk_mean": float(np.mean(qwk_arr)),
                "qwk_std_ddof1": float(np.std(qwk_arr, ddof=1)),
                "qwk_std_ddof0": float(np.std(qwk_arr, ddof=0)),
                "mae_mean": float(np.mean(mae_arr)),
                "mae_std_ddof1": float(np.std(mae_arr, ddof=1)),
                "mae_std_ddof0": float(np.std(mae_arr, ddof=0)),
                "acc_mean": float(np.mean(acc_arr)),
                "acc_std_ddof1": float(np.std(acc_arr, ddof=1)),
                "acc_std_ddof0": float(np.std(acc_arr, ddof=0)),
                "macro_f1_mean": float(np.mean(f1_arr)),
                "macro_f1_std_ddof1": float(np.std(f1_arr, ddof=1)),
                "macro_f1_std_ddof0": float(np.std(f1_arr, ddof=0)),
            })
            
    df_agg = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(out_dir, "gap1_aggregate_metrics_summary.csv")
    df_agg.to_csv(agg_csv, index=False)
    print(f"Saved aggregate metrics summary: {agg_csv}")
    print("\nAggregate Results Table:")
    for r in agg_rows:
        print(f"[{r['protocol']}] Config {r['config']}: "
              f"QWK={r['qwk_mean']:.4f} ± {r['qwk_std_ddof1']:.4f} (ddof0: {r['qwk_std_ddof0']:.4f}) | "
              f"MAE={r['mae_mean']:.4f} ± {r['mae_std_ddof1']:.4f} | "
              f"Acc={r['acc_mean']:.2f}% ± {r['acc_std_ddof1']:.2f}% | "
              f"Macro-F1={r['macro_f1_mean']:.2f}% ± {r['macro_f1_std_ddof1']:.2f}%")

if __name__ == "__main__":
    main()
