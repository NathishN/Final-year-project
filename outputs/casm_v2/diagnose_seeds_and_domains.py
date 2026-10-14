import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import torch
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, precision_recall_fscore_support

from context_audit_train import (
    set_seed, GPUDataset, AuditModel, CORALLoss,
    eval_ds, compute_metrics, MODEL_CONFIGS
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

data_dict = torch.load("outputs/features/phase4_dataset.pt", map_location="cpu", weights_only=False)
print("Loaded feature dataset.")

CONFIGS = {
    "A": {"name": "Image Only", "use_species": False, "use_ood": False, "use_grade": False, "use_unc": False},
    "B": {"name": "Image + Species", "use_species": True, "use_ood": False, "use_grade": False, "use_unc": False},
    "D": {"name": "Image + Species+OOD", "use_species": True, "use_ood": True, "use_grade": False, "use_unc": False},
}

SEEDS_TO_ANALYZE = [19, 31, 42, 23] # 19 and 31 (anomalous), 42 (nominal), 23 (best baseline)

def train_and_diagnose(cfg_key, seed):
    cfg = CONFIGS[cfg_key]
    set_seed(seed)
    tr = GPUDataset(data_dict, "train", device)
    va = GPUDataset(data_dict, "val", device)
    te = GPUDataset(data_dict, "test", device)
    
    counts = np.bincount(tr.tgt.cpu().numpy(), minlength=5)
    cw = torch.from_numpy(len(tr.tgt)/(5.0*np.maximum(counts,1))).float().to(device)
    model = AuditModel(cfg).to(device)
    crit = CORALLoss(num_classes=5, class_weights=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-5)
    
    best_mae, best_state = 999.0, None
    history = []
    
    for ep in range(1, 16):
        model.train()
        perm = torch.randperm(tr.n, device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, tr.n, 2048):
            idx = perm[i:i+2048]
            vis, spec, grade, ood, unc, tgt = tr.get_batch(idx)
            opt.zero_grad()
            loss = crit(model(vis, spec, grade, ood, unc), tgt)
            loss.backward(); opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()
        
        # Eval val
        model.eval()
        with torch.no_grad():
            val_p = model.predict_rank(model(va.vis, va.spec, va.grade, va.ood, va.unc))
            val_mae = torch.mean(torch.abs(val_p.float() - va.tgt.float())).item()
        
        history.append({"epoch": ep, "train_loss": total_loss/n_batches, "val_mae": val_mae})
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(te.vis, te.spec, te.grade, te.ood, te.unc)
        test_preds = model.predict_rank(test_logits).cpu().numpy()
        test_tgts = te.tgt.cpu().numpy()
        
    metrics = compute_metrics(test_tgts, test_preds)
    cm = confusion_matrix(test_tgts, test_preds, labels=[0, 1, 2, 3, 4])
    pred_dist = Counter(test_preds.tolist())
    
    # Per-domain metrics on test set
    test_domains = data_dict["domains"][data_dict["splits"] == "test"]
    domain_results = {}
    for dom in np.unique(test_domains):
        mask_d = (test_domains == dom)
        d_tgts = test_tgts[mask_d]
        d_preds = test_preds[mask_d]
        d_m = compute_metrics(d_tgts, d_preds)
        domain_results[dom] = {
            "N": int(len(d_tgts)),
            "Accuracy": d_m["Accuracy"],
            "MAE": d_m["MAE"],
            "QWK": d_m["QWK"],
            "pred_dist": Counter(d_preds.tolist())
        }
        
    return {
        "cfg": cfg_key,
        "seed": seed,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "pred_dist": dict(pred_dist),
        "domain_results": domain_results,
        "best_val_mae": best_mae,
        "history": history
    }

print("Running diagnosis across seeds...")
results = {}
for s in SEEDS_TO_ANALYZE:
    for cfg in ["A", "D"]:
        key = f"{cfg}_seed{s}"
        print(f"Training {key}...")
        res = train_and_diagnose(cfg, s)
        results[key] = res
        print(f"  {key}: QWK={res['metrics']['QWK']:.4f}, MAE={res['metrics']['MAE']:.4f}, Acc={res['metrics']['Accuracy']:.2f}%, Pred dist={res['pred_dist']}")

with open("outputs/casm_v2/phase45_stability_diagnostics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved outputs/casm_v2/phase45_stability_diagnostics.json")
