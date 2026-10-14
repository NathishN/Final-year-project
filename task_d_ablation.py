"""
Phase 4.4 Task D: Controlled 4-way Species/OOD ablation.
Configurations: A=Image-Only, B=Image+Species, C=Image+OOD, D=Image+Species+OOD
Identical seeds, splits, features, training budget, checkpoint selection.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
import torch
from context_audit_train import (
    set_seed, GPUDataset, AuditModel, CORALLoss,
    eval_ds, eval_ds_fast, compute_metrics
)

os.makedirs("outputs/casm_v2", exist_ok=True)
AUDIT_DIR = "outputs/casm_v2/ablation_ckpts"
os.makedirs(AUDIT_DIR, exist_ok=True)

SEEDS   = [42, 123, 777, 999, 2025, 1, 2, 3, 5, 7,
           11, 13, 17, 19, 23, 31, 37, 41, 43, 47]
EPOCHS  = 15
LR      = 2e-3
WD      = 1e-4
BATCH   = 2048

CONFIGS = {
    "A": {"name": "Image Only",          "use_species": False, "use_ood": False, "use_grade": False, "use_unc": False},
    "B": {"name": "Image + Species",     "use_species": True,  "use_ood": False, "use_grade": False, "use_unc": False},
    "C": {"name": "Image + OOD",         "use_species": False, "use_ood": True,  "use_grade": False, "use_unc": False},
    "D": {"name": "Image + Species+OOD", "use_species": True,  "use_ood": True,  "use_grade": False, "use_unc": False},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Configurations: {list(CONFIGS.keys())}")
print(f"Seeds: {len(SEEDS)}")
print(f"Total runs: {len(CONFIGS)*len(SEEDS)}")
print()

def train_one(cfg_key, seed, data_dict):
    cfg = CONFIGS[cfg_key]
    set_seed(seed)
    tr = GPUDataset(data_dict, "train", device)
    va = GPUDataset(data_dict, "val",   device)
    te = GPUDataset(data_dict, "test",  device)
    counts = np.bincount(tr.tgt.cpu().numpy(), minlength=5)
    cw     = torch.from_numpy(len(tr.tgt)/(5.0*np.maximum(counts,1))).float().to(device)
    model  = AuditModel(cfg).to(device)
    crit   = CORALLoss(num_classes=5, class_weights=cw)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    best_mae, best_state = 999.0, None
    for ep in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(tr.n, device=device)
        for i in range(0, tr.n, BATCH):
            idx = perm[i:i+BATCH]
            vis, spec, grade, ood, unc, tgt = tr.get_batch(idx)
            opt.zero_grad()
            loss = crit(model(vis, spec, grade, ood, unc), tgt)
            loss.backward(); opt.step()
        sched.step()
        val_mae, _ = eval_ds_fast(model, va)
        if val_mae < best_mae:
            best_mae, best_state = val_mae, {k:v.cpu() for k,v in model.state_dict().items()}
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    metrics, _, _ = eval_ds(model, te)
    return metrics

# --- Train all 4 configs x 20 seeds ---
data_dict = torch.load("outputs/features/phase4_dataset.pt", map_location="cpu", weights_only=False)
print(f"Dataset loaded: {len(data_dict['targets'])} samples")

rows = []
for cfg_key, cfg in CONFIGS.items():
    print(f"\n[Config {cfg_key}] {cfg['name']}")
    for seed in SEEDS:
        m = train_one(cfg_key, seed, data_dict)
        rows.append({"Config": cfg_key, "Name": cfg["name"], "Seed": seed, **m})
        print(f"  seed={seed:4d}  Acc={m['Accuracy']:.2f}%  QWK={m['QWK']:.4f}  MAE={m['MAE']:.4f}", flush=True)

df_raw = pd.DataFrame(rows)
df_raw.to_csv("outputs/casm_v2/phase44_ablation_raw.csv", index=False)
print("\nSaved phase44_ablation_raw.csv")

# --- Aggregate ---
agg_rows = []
for cfg_key, cfg in CONFIGS.items():
    sub = df_raw[df_raw.Config==cfg_key]
    for m in ["QWK","MAE","Accuracy","OBO_Acc","Macro_F1"]:
        pass
    agg_rows.append({
        "Config": cfg_key, "Name": cfg["name"],
        "QWK_mean":  round(sub.QWK.mean(),4),  "QWK_std":  round(sub.QWK.std(ddof=1),4),
        "MAE_mean":  round(sub.MAE.mean(),4),  "MAE_std":  round(sub.MAE.std(ddof=1),4),
        "Acc_mean":  round(sub.Accuracy.mean(),4), "Acc_std": round(sub.Accuracy.std(ddof=1),4),
        "OBO_mean":  round(sub.OBO_Acc.mean(),4), "OBO_std": round(sub.OBO_Acc.std(ddof=1),4),
        "F1_mean":   round(sub.Macro_F1.mean(),4), "F1_std":  round(sub.Macro_F1.std(ddof=1),4),
    })
df_agg = pd.DataFrame(agg_rows)
df_agg.to_csv("outputs/casm_v2/phase44_ablation_results.csv", index=False)
print("Saved phase44_ablation_results.csv")
print()
print(df_agg[["Config","Name","QWK_mean","QWK_std","MAE_mean","OBO_mean"]].to_string(index=False))
print()

# --- Paired comparisons ---
METRICS = ["QWK","MAE","Accuracy","OBO_Acc","Macro_F1"]
comparisons = [
    ("D","A","Primary: Species+OOD vs Image-Only"),
    ("B","A","Secondary: Species vs Image-Only"),
    ("C","A","Secondary: OOD vs Image-Only"),
    ("D","B","Secondary: Species+OOD vs Species-Only"),
    ("D","C","Secondary: Species+OOD vs OOD-Only"),
]

sig_rows = []
for (cfg1, cfg0, label) in comparisons:
    s1 = df_raw[df_raw.Config==cfg1].sort_values("Seed").reset_index(drop=True)
    s0 = df_raw[df_raw.Config==cfg0].sort_values("Seed").reset_index(drop=True)
    assert list(s1.Seed)==list(s0.Seed), f"Seed mismatch in {cfg1} vs {cfg0}"
    row = {"Comparison": label, "Config1": cfg1, "Config0": cfg0}
    for m in METRICS:
        v1,v0 = s1[m].values, s0[m].values
        dv    = v1-v0
        t,p   = stats.ttest_rel(v1,v0)
        delta = float(np.mean(dv))
        sd    = float(np.std(dv,ddof=1))
        pl    = float(np.sqrt((np.std(v0,ddof=1)**2+np.std(v1,ddof=1)**2)/2))
        cd    = delta/pl if pl>0 else 0.0
        se    = sd/np.sqrt(len(v1))
        ci_lo = delta - stats.t.ppf(0.975, df=len(v1)-1)*se
        ci_hi = delta + stats.t.ppf(0.975, df=len(v1)-1)*se
        row[f"{m}_delta"] = round(delta,4)
        row[f"{m}_p"]     = round(float(p),4)
        row[f"{m}_cohen"] = round(cd,4)
        row[f"{m}_ci_lo"] = round(ci_lo,4)
        row[f"{m}_ci_hi"] = round(ci_hi,4)
    sig_rows.append(row)

df_sig = pd.DataFrame(sig_rows)

# --- Holm-Bonferroni on secondary comparisons for QWK ---
# Primary: D vs A (1 test)
# Secondary family: B vs A, C vs A, D vs B, D vs C (4 tests)
sec_rows = df_sig[df_sig.Comparison.str.startswith("Secondary")]
sec_ps   = sec_rows["QWK_p"].values.astype(float)
order    = np.argsort(sec_ps)
adj_ps   = np.zeros(len(sec_ps))
for rank, idx in enumerate(order):
    adj_ps[idx] = min(1.0, sec_ps[idx]*(len(sec_ps)-rank))
for i in range(len(order)-1, 0, -1):
    adj_ps[order[i-1]] = min(adj_ps[order[i-1]], adj_ps[order[i]])

print("=== PRIMARY COMPARISON (D vs A) ===")
prim = df_sig[df_sig.Config1=="D"][df_sig.Config0=="A"].iloc[0]
print(f"  QWK: delta={prim.QWK_delta:+.4f}  p={prim.QWK_p:.4f}  Cohen_d={prim.QWK_cohen:+.4f}  CI=[{prim.QWK_ci_lo:+.4f},{prim.QWK_ci_hi:+.4f}]")
print(f"  MAE: delta={prim.MAE_delta:+.4f}  p={prim.MAE_p:.4f}")
print(f"  OBO: delta={prim.OBO_Acc_delta:+.4f}  p={prim.OBO_Acc_p:.4f}")
print()
print("=== SECONDARY COMPARISONS (Holm-Bonferroni corrected) ===")
for i,(_, r) in enumerate(sec_rows.iterrows()):
    print(f"  {r.Comparison}: QWK_delta={r.QWK_delta:+.4f}  raw_p={r.QWK_p:.4f}  holm_adj_p={adj_ps[i]:.4f}")

df_sig.to_csv("outputs/casm_v2/phase44_ablation_significance.csv", index=False)
print("\nSaved phase44_ablation_significance.csv")
print("TASK D COMPLETE")
