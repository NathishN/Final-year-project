"""
Phase 5.2 -- Session-held-out RETRAINING experiment (new evidence, not a reproduction).

Why: the earlier Phase 5.2 "holdout" re-scored split_v2 models after moving session
20200730 into test. Those models had trained on 147 of that session's images, so the
evaluation contained training samples. This script retrains every head with the whole
session excluded from train, using the identical protocol of phase50_stage5_confirmatory.py
(15 epochs, AdamW lr=1e-3, wd=1e-4, batch 2048, Effective-Number weights from train only,
checkpoint selected on validation QWK). Historical checkpoints are NOT touched.
"""
import os, sys, io, json
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = r"C:\fruit_dataset"
sys.path.insert(0, ROOT)
import phase50_stage5_confirmatory as conf
from phase50_casm_v2_model import compute_effective_num_weights

OUT = os.path.join(ROOT, "outputs", "phase52_final_validation", "session_holdout_retrain")
conf.CKPT_DIR = os.path.join(OUT, "checkpoints")
conf.PREDS_DIR = os.path.join(OUT, "predictions")
os.makedirs(conf.CKPT_DIR, exist_ok=True)
os.makedirs(conf.PREDS_DIR, exist_ok=True)

manifest = pd.read_csv(os.path.join(ROOT, "outputs", "phase52_final_validation", "session_holdout_manifest.csv"))
data = conf.load_dataset()
assert list(manifest["image_path"]) == list(data["manifest"]["image_path"]), "manifest/feature order mismatch"
splits = manifest["split"].values
data["train_idx"] = np.where(splits == "train")[0]
data["val_idx"] = np.where(splits == "val")[0]
data["test_idx"] = np.where(splits == "test")[0]

# Leakage guard: no session may appear in more than one partition
sess = manifest["session_id"].astype(str).values
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    inter = set(sess[splits == a]) & set(sess[splits == b])
    assert not inter, f"session overlap {a}/{b}: {inter}"
print(f"Train {len(data['train_idx'])}, Val {len(data['val_idx'])}, Test {len(data['test_idx'])}; 0 cross-split sessions")

cw = compute_effective_num_weights(data["tgt"][data["train_idx"]], beta=0.999, num_classes=5).to(conf.DEVICE)

records = []
for cfg in conf.CONFIGS:
    for seed in conf.SEEDS:
        r = conf.run_single_experiment(data, cfg, seed, cw, conf.DEVICE, epochs=15, lr=1e-3, batch_size=2048)
        records.append(r)
        print(f"  {cfg} seed {seed:4d}: val_qwk={r['val_qwk']:.4f} test_qwk={r['test_qwk']:.4f} mae={r['test_mae']:.4f}", flush=True)

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT, "per_seed_results.csv"), index=False)

summary = df.groupby("config").agg(
    n=("seed", "count"), qwk_mean=("test_qwk", "mean"), qwk_sd=("test_qwk", "std"),
    mae_mean=("test_mae", "mean"), acc_mean=("test_acc", "mean"),
    macro_f1_mean=("test_macro_f1", "mean"), g2_recall_mean=("rec_g2", "mean")).reset_index()
summary.to_csv(os.path.join(OUT, "summary_by_config.csv"), index=False)

piv = df.pivot(index="seed", columns="config", values="test_qwk")
rows, raw_p = [], []
for c in ["B", "C", "D"]:
    d = (piv[c] - piv["A"]).values
    t = stats.ttest_1samp(d, 0.0)
    w = stats.wilcoxon(d, zero_method="wilcox")
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
    rows.append({"contrast": f"{c}-A", "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
                 "ci_low": ci[0], "ci_high": ci[1], "dz": d.mean() / d.std(ddof=1),
                 "t": t.statistic, "p_t": t.pvalue, "W": w.statistic, "p_w": w.pvalue,
                 "n_positive": int((d > 0).sum())})
    raw_p.append(t.pvalue)
order = np.argsort(raw_p); m = len(raw_p); adj = [0.0] * m; running = 0.0
for k, i in enumerate(order):
    running = max(running, min(1.0, raw_p[i] * (m - k))); adj[i] = running
for i, r in enumerate(rows):
    r["p_holm"] = adj[i]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "paired_stats.csv"), index=False)
print(summary.to_string(index=False))
print(pd.DataFrame(rows).to_string(index=False))
