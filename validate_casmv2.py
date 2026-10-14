"""
validate_casmv2.py
==================
CASM v2 Independent Validation
Trains Image-Only (baseline) and CASM v2 (Species+OOD) with N=20 seeds.
Performs paired t-test with Cohen d effect size.
Generates: casmv2_validation_report.md + casmv2_validation_results.csv
"""

import os, sys, json, warnings, random
import numpy as np
import pandas as pd
from scipy import stats
import torch
warnings.filterwarnings("ignore")

from context_audit_train import (
    set_seed, GPUDataset, AuditModel, CORALLoss,
    eval_ds, eval_ds_fast, compute_metrics
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "outputs/features/phase4_dataset.pt"
OUT_DIR      = "outputs/phase4/casmv2"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS_V2 = [42, 123, 777, 999, 2025,
            1, 2, 3, 5, 7,
            11, 13, 17, 19, 23,
            31, 37, 41, 43, 47]          # N=20 seeds

EPOCHS   = 15
LR       = 2e-3
WD       = 1e-4
BATCH    = 2048

BASELINE_CFG = {"name": "Image Only (Baseline)",
                "use_species": False, "use_ood": False,
                "use_grade":   False, "use_unc": False}

CASMV2_CFG   = {"name": "CASM v2 (Species+OOD)",
                "use_species": True,  "use_ood": True,
                "use_grade":   False, "use_unc": False}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
print(f"Seeds  : {len(SEEDS_V2)}")
print(f"Epochs : {EPOCHS}")


# ── Training helper ───────────────────────────────────────────────────────────
def train_one(cfg, seed, data_dict):
    set_seed(seed)
    tr = GPUDataset(data_dict, "train", device)
    va = GPUDataset(data_dict, "val",   device)
    te = GPUDataset(data_dict, "test",  device)

    counts = np.bincount(tr.tgt.cpu().numpy(), minlength=5)
    cw     = torch.from_numpy(len(tr.tgt) / (5.0 * np.maximum(counts, 1))).float().to(device)

    model  = AuditModel(cfg).to(device)
    crit   = CORALLoss(num_classes=5, class_weights=cw)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    best_mae, best_state = 999.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(tr.n, device=device)
        for i in range(0, tr.n, BATCH):
            idx = perm[i:i+BATCH]
            vis, spec, grade, ood, unc, tgt = tr.get_batch(idx)
            opt.zero_grad()
            loss = crit(model(vis, spec, grade, ood, unc), tgt)
            loss.backward()
            opt.step()
        sched.step()
        val_mae, _ = eval_ds_fast(model, va)
        if val_mae < best_mae:
            best_mae   = val_mae
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    metrics, _, _ = eval_ds(model, te)
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────
print(f"\nLoading dataset: {DATASET_PATH}")
data_dict = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
print(f"Samples: {len(data_dict['targets'])}")

rows = []
for label, cfg in [("Baseline", BASELINE_CFG), ("CASMv2", CASMV2_CFG)]:
    print(f"\n{'='*60}")
    print(f"Training: {cfg['name']}")
    print(f"{'='*60}")
    for seed in SEEDS_V2:
        m = train_one(cfg, seed, data_dict)
        rows.append({"Model": label, "Name": cfg["name"], "Seed": seed, **m})
        print(f"  seed={seed:4d}  Acc={m['Accuracy']:.2f}%  "
              f"QWK={m['QWK']:.4f}  MAE={m['MAE']:.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("casmv2_validation_results.csv", index=False)
df.to_csv(os.path.join(OUT_DIR, "casmv2_per_seed.csv"), index=False)
print("\nSaved casmv2_validation_results.csv")


# ── Statistics ────────────────────────────────────────────────────────────────
base  = df[df.Model == "Baseline"]
v2    = df[df.Model == "CASMv2"]

results = {}
for metric in ["QWK", "MAE", "Accuracy", "Macro_F1", "OBO_Acc"]:
    bv, cv = base[metric].values, v2[metric].values
    t, p   = stats.ttest_rel(bv, cv)
    d_mean = float(np.mean(cv) - np.mean(bv))
    pooled = float(np.sqrt((np.std(bv, ddof=1)**2 + np.std(cv, ddof=1)**2) / 2))
    cohend = d_mean / pooled if pooled > 0 else 0.0
    results[metric] = {
        "base_mean": round(float(np.mean(bv)), 4),
        "base_std":  round(float(np.std(bv, ddof=1)), 4),
        "v2_mean":   round(float(np.mean(cv)), 4),
        "v2_std":    round(float(np.std(cv, ddof=1)), 4),
        "delta":     round(d_mean, 4),
        "t":         round(float(t), 4),
        "p":         round(float(p), 4),
        "cohen_d":   round(cohend, 4),
        "sig":       "YES" if p < 0.05 else "NO",
        "direction": ("BETTER" if (d_mean > 0 if metric != "MAE" else d_mean < 0) else "WORSE"),
    }

# ── Report ────────────────────────────────────────────────────────────────────
qwk_r  = results["QWK"]
mae_r  = results["MAE"]
acc_r  = results["Accuracy"]
sig_ok = qwk_r["sig"] == "YES" and qwk_r["direction"] == "BETTER"

# Cohen d interpretation
def interpret_d(d):
    a = abs(d)
    return "LARGE" if a >= 0.8 else ("MEDIUM" if a >= 0.5 else ("SMALL" if a >= 0.2 else "NEGLIGIBLE"))

lines = [
    "# CASM v2 Independent Validation Report",
    "",
    f"**Model:** CASM v2 — Species-Aware + OOD Concatenation Fusion",
    f"**Baseline:** Image-Only (ConvNeXt-Tiny → CORAL head)",
    f"**Validation:** Paired t-test, N={len(SEEDS_V2)} seeds, α=0.05",
    f"**Dataset:** 56,036 samples × 5 fruit species",
    "",
    "---",
    "",
    "## Primary Results",
    "",
    "| Metric | Baseline (Image Only) | CASM v2 (Species+OOD) | Δ | p-value | Cohen d | Significant? |",
    "|--------|-----------------------|-----------------------|---|---------|---------|--------------|",
]
for metric in ["QWK", "MAE", "Accuracy", "Macro_F1", "OBO_Acc"]:
    r = results[metric]
    sig_str = f"✅ YES" if r["sig"] == "YES" and r["direction"] == "BETTER" else \
              f"❌ HARMFUL" if r["sig"] == "YES" and r["direction"] == "WORSE" else \
              f"— NS"
    lines.append(
        f"| {metric} | {r['base_mean']:.4f}±{r['base_std']:.4f} | "
        f"{r['v2_mean']:.4f}±{r['v2_std']:.4f} | "
        f"{r['delta']:+.4f} | {r['p']:.4f} | "
        f"{r['cohen_d']:+.4f} ({interpret_d(r['cohen_d'])}) | {sig_str} |"
    )

verdict = ("✅ VALIDATED" if sig_ok else
           ("📈 POSITIVE TREND (Not significant)" if qwk_r["delta"] > 0 else
            "❌ NOT VALIDATED"))

lines += [
    "",
    "---",
    "",
    "## Verdict",
    "",
    f"### Overall: {verdict}",
    "",
    f"- **QWK improvement:** ΔQWK={qwk_r['delta']:+.4f} (p={qwk_r['p']:.4f}, Cohen d={qwk_r['cohen_d']:+.4f} [{interpret_d(qwk_r['cohen_d'])}])",
    f"- **MAE improvement:** ΔMAE={mae_r['delta']:+.4f} (p={mae_r['p']:.4f})",
    f"- **Accuracy change:** ΔAcc={acc_r['delta']:+.4f}% (p={acc_r['p']:.4f})",
    "",
    "---",
    "",
    "## Per-Seed QWK Summary",
    "",
    "| Seed | Baseline QWK | CASM v2 QWK | Δ |",
    "|------|-------------|-------------|---|",
]
for seed in SEEDS_V2:
    b_row = base[base.Seed == seed]
    v_row = v2[v2.Seed   == seed]
    if len(b_row) and len(v_row):
        bq = float(b_row.QWK.values[0])
        vq = float(v_row.QWK.values[0])
        win = "▲" if vq > bq else ("▽" if vq < bq else "=")
        lines.append(f"| {seed} | {bq:.4f} | {vq:.4f} | {vq-bq:+.4f} {win} |")

base_wins = sum(1 for s in SEEDS_V2
                if len(base[base.Seed==s]) and len(v2[v2.Seed==s]) and
                float(base[base.Seed==s].QWK.values[0]) > float(v2[v2.Seed==s].QWK.values[0]))
v2_wins   = len(SEEDS_V2) - base_wins

lines += [
    "",
    f"**CASM v2 wins:** {v2_wins}/{len(SEEDS_V2)} seeds",
    f"**Baseline wins:** {base_wins}/{len(SEEDS_V2)} seeds",
    "",
    "---",
    "",
    "## Publication Framing",
    "",
]

if sig_ok:
    lines += [
        "**CASM v2 is statistically validated.** Replace full CASM claim with:",
        "> 'Species-Aware CASM v2 achieves statistically significant QWK improvement",
        "> over the Image-Only baseline (ΔQWK={:+.4f}, p={:.4f}, d={:.4f}).'".format(
            qwk_r["delta"], qwk_r["p"], qwk_r["cohen_d"]),
        "",
        "**Title confirmation:** Use 'Species-Aware Severity Modeling' — empirically validated.",
    ]
elif qwk_r["delta"] > 0:
    lines += [
        "**Positive trend confirmed, statistical significance not reached at α=0.05.**",
        f"> ΔQWK={qwk_r['delta']:+.4f}, p={qwk_r['p']:.4f}, N={len(SEEDS_V2)} seeds.",
        "",
        "**Publishing options:**",
        "1. Report as positive trend + honest power analysis (recommended for negative-finding framing)",
        "2. Increase N to ≥50 seeds for definitive confirmation",
        "3. Reframe: 'CASM v2 shows consistent improvement across {}/{} seeds".format(v2_wins, len(SEEDS_V2)) + " with positive effect size.'",
    ]
else:
    lines += [
        "**CASM v2 does not improve over baseline.** Use Image-Only framing.",
        "Publishable as rigorous negative finding with full Context Contribution Audit.",
    ]

lines += [
    "",
    "---",
    "",
    "## Comparison: Phase 4 Audit vs CASM v2 Validation",
    "",
    "| Experiment | Model | N seeds | QWK Mean±Std |",
    "|-----------|-------|---------|--------------|",
    f"| Phase 4.3 Audit | Image Only (A) | 5 | 0.7222±0.0210 |",
    f"| Phase 4.3 Audit | Full CASM (I)  | 5 | 0.7405±0.0370 |",
    f"| Phase 4.3 Audit | Species+OOD (F)| 5 | 0.7385±0.0200 |",
    f"| **CASM v2 Validation** | **Baseline**   | **{len(SEEDS_V2)}** | **{qwk_r['base_mean']:.4f}±{qwk_r['base_std']:.4f}** |",
    f"| **CASM v2 Validation** | **CASM v2**    | **{len(SEEDS_V2)}** | **{qwk_r['v2_mean']:.4f}±{qwk_r['v2_std']:.4f}** |",
    "",
    "---",
    "",
    "## Files Generated",
    "- `casmv2_validation_results.csv` — per-seed metrics for both models",
    "- `casmv2_validation_report.md` — this report",
    f"- `{OUT_DIR}/casmv2_per_seed.csv` — backup copy",
]

report_text = "\n".join(lines)
open("casmv2_validation_report.md", "w", encoding="utf-8").write(report_text)
print("\nSaved casmv2_validation_report.md")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CASM v2 VALIDATION SUMMARY")
print("="*60)
print(f"Baseline QWK : {qwk_r['base_mean']:.4f} ± {qwk_r['base_std']:.4f}")
print(f"CASM v2  QWK : {qwk_r['v2_mean']:.4f} ± {qwk_r['v2_std']:.4f}")
print(f"Delta QWK    : {qwk_r['delta']:+.4f}")
print(f"p-value      : {qwk_r['p']:.4f}  ({'SIGNIFICANT' if qwk_r['sig'] == 'YES' else 'NOT SIGNIFICANT'} at alpha=0.05)")
print(f"Cohen d      : {qwk_r['cohen_d']:+.4f}  [{interpret_d(qwk_r['cohen_d'])} effect]")
print(f"CASM v2 wins : {v2_wins}/{len(SEEDS_V2)} seeds")
print(f"Verdict      : {verdict}")
print("="*60)
