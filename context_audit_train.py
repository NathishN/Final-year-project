"""
context_audit_train.py
======================
Context Contribution Audit Training Engine.
Trains Models A-I with selective context branches across multiple seeds.

NEW audit labeling (different from Phase 4 A-E):
  A: Image Only
  B: Image + Species
  C: Image + OOD Context
  D: Image + Grade Context
  E: Image + Uncertainty Context
  F: Image + Species + OOD
  G: Image + Species + Grade
  H: Image + OOD + Grade
  I: Full CASM (Species + OOD + Grade + Uncertainty)
"""

import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_recall_fscore_support


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


MODEL_CONFIGS = {
    "A": {"name": "Image Only",              "use_species": False, "use_ood": False, "use_grade": False, "use_unc": False},
    "B": {"name": "Image + Species",         "use_species": True,  "use_ood": False, "use_grade": False, "use_unc": False},
    "C": {"name": "Image + OOD",             "use_species": False, "use_ood": True,  "use_grade": False, "use_unc": False},
    "D": {"name": "Image + Grade",           "use_species": False, "use_ood": False, "use_grade": True,  "use_unc": False},
    "E": {"name": "Image + Uncertainty",     "use_species": False, "use_ood": False, "use_grade": False, "use_unc": True},
    "F": {"name": "Image + Species + OOD",   "use_species": True,  "use_ood": True,  "use_grade": False, "use_unc": False},
    "G": {"name": "Image + Species + Grade", "use_species": True,  "use_ood": False, "use_grade": True,  "use_unc": False},
    "H": {"name": "Image + OOD + Grade",     "use_species": False, "use_ood": True,  "use_grade": True,  "use_unc": False},
    "I": {"name": "Full CASM",               "use_species": True,  "use_ood": True,  "use_grade": True,  "use_unc": True},
}
SEEDS = [42, 123, 777, 999, 2025]


class GPUDataset:
    def __init__(self, data_dict, mask, device="cuda"):
        m = (mask == True) if isinstance(mask, torch.Tensor) else (data_dict["splits"] == mask)
        self.vis   = data_dict["vis_feats"][m].to(device)
        self.spec  = data_dict["species_ids"][m].to(device)
        self.grade = data_dict["grade_ctx"][m].to(device)
        self.ood   = data_dict["ood_ctx"][m].to(device)
        self.unc   = data_dict["unc_ctx"][m].to(device)
        self.tgt   = data_dict["targets"][m].to(device)
        self.n     = len(self.tgt)

    def get_batch(self, indices):
        return (self.vis[indices], self.spec[indices], self.grade[indices],
                self.ood[indices], self.unc[indices], self.tgt[indices])


class CORALHead(nn.Module):
    def __init__(self, in_features=128, num_classes=5):
        super().__init__()
        self.num_tasks = num_classes - 1
        self.weight = nn.Parameter(torch.Tensor(1, in_features))
        self.bias   = nn.Parameter(torch.zeros(self.num_tasks))
        nn.init.kaiming_uniform_(self.weight, a=2.0**0.5)
    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias
    def predict_rank(self, logits):
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)


class AuditModel(nn.Module):
    """Simple concatenation / linear projection fusion to isolate individual context contributions."""
    def __init__(self, cfg, num_species=5, num_classes=5):
        super().__init__()
        self.cfg = cfg
        self.vis_proj = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU()
        )
        ctx_dim = 0
        if cfg["use_species"]: self.species_embed = nn.Embedding(num_species, 16); ctx_dim += 16
        if cfg["use_ood"]:     ctx_dim += 2
        if cfg["use_grade"]:   ctx_dim += 4
        if cfg["use_unc"]:     ctx_dim += 2
        fused = 256 + ctx_dim
        self.fc = nn.Sequential(
            nn.Linear(fused, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128),   nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1)
        )
        self.head = CORALHead(128, num_classes)

    def forward(self, vis, species, grade, ood, unc):
        parts = [self.vis_proj(vis)]
        if self.cfg["use_species"]: parts.append(self.species_embed(species))
        if self.cfg["use_ood"]:     parts.append(ood)
        if self.cfg["use_grade"]:   parts.append(grade)
        if self.cfg["use_unc"]:     parts.append(unc)
        return self.head(self.fc(torch.cat(parts, dim=1)))

    def predict_rank(self, logits):
        return self.head.predict_rank(logits)


class CORALLoss(nn.Module):
    def __init__(self, num_classes=5, class_weights=None):
        super().__init__()
        self.num_tasks = num_classes - 1
        self.cw = class_weights
    def forward(self, logits, targets):
        lvl = torch.arange(self.num_tasks, device=targets.device).unsqueeze(0)
        bt  = (targets.unsqueeze(1) > lvl).float()
        L   = F.binary_cross_entropy_with_logits(logits, bt, reduction="none")
        if self.cw is not None: L = L * self.cw[targets].unsqueeze(1)
        return L.mean()


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    try:    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except: qwk = 0.0
    obo = float(np.mean(np.abs(y_true - y_pred) <= 1)) * 100.0
    _, _, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",    zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"Accuracy": round(acc, 4), "MAE": round(mae, 4), "QWK": round(qwk, 4),
            "OBO_Acc":  round(obo, 4), "Macro_F1": round(f1m*100, 4), "Weighted_F1": round(f1w*100, 4)}


@torch.no_grad()
def eval_ds_fast(model, ds, batch=1024):
    model.eval()
    preds, targets = [], []
    for i in range(0, ds.n, batch):
        idx = torch.arange(i, min(i + batch, ds.n), device=ds.vis.device)
        vis, spec, grade, ood, unc, tgt = ds.get_batch(idx)
        p = model.predict_rank(model(vis, spec, grade, ood, unc))
        preds.append(p)
        targets.append(tgt)
    p_cat = torch.cat(preds)
    t_cat = torch.cat(targets)
    mae = torch.mean(torch.abs(p_cat.float() - t_cat.float())).item()
    acc = (p_cat == t_cat).float().mean().item() * 100.0
    return mae, acc


@torch.no_grad()
def eval_ds(model, ds, batch=1024):
    model.eval()
    preds, targets = [], []
    for i in range(0, ds.n, batch):
        idx = torch.arange(i, min(i + batch, ds.n), device=ds.vis.device)
        vis, spec, grade, ood, unc, tgt = ds.get_batch(idx)
        p = model.predict_rank(model(vis, spec, grade, ood, unc)).cpu().numpy()
        preds.extend(p)
        targets.extend(tgt.cpu().numpy())
    return compute_metrics(targets, preds), np.array(targets), np.array(preds)


def train_model(model_id, seed, data_dict, device, epochs=12, lr=2e-3, wd=1e-4,
                batch=2048, output_dir="outputs/phase4/audit"):
    set_seed(seed)
    cfg = MODEL_CONFIGS[model_id]
    os.makedirs(output_dir, exist_ok=True)
    
    train_ds = GPUDataset(data_dict, "train", device=device)
    val_ds   = GPUDataset(data_dict, "val",   device=device)
    test_ds  = GPUDataset(data_dict, "test",  device=device)

    tgt_arr = train_ds.tgt.cpu().numpy()
    counts  = np.bincount(tgt_arr, minlength=5)
    cw      = torch.from_numpy(len(tgt_arr) / (5.0 * np.maximum(counts, 1))).float().to(device)

    model     = AuditModel(cfg).to(device)
    criterion = CORALLoss(num_classes=5, class_weights=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_mae = 999.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(train_ds.n, device=device)
        for i in range(0, train_ds.n, batch):
            idx = perm[i:i + batch]
            vis, spec, grade, ood, unc, tgt = train_ds.get_batch(idx)
            optimizer.zero_grad()
            loss = criterion(model(vis, spec, grade, ood, unc), tgt)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_mae, _ = eval_ds_fast(model, val_ds)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    ckpt_path = os.path.join(output_dir, f"model_{model_id}_seed{seed}_best.pth")
    torch.save({"state_dict": best_state, "cfg": cfg, "val_mae": best_val_mae}, ckpt_path)

    test_m, tgts, preds = eval_ds(model, test_ds)
    return test_m, tgts, preds


if __name__ == "__main__":
    print("context_audit_train.py ready. Models:", list(MODEL_CONFIGS.keys()))

