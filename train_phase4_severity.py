"""
train_phase4_severity.py
========================
Training Pipeline for Phase 4 Context-Aware Severity Modeling (CASM).

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Features:
- Supports Ablation Modes: 'A' (Img Only), 'B' (+Spec), 'C' (+Grade), 'D' (+OOD), 'E' (Full CASM)
- Supports Loss Functions: 'coral', 'corn', 'ce', 'focal'
- Saves best model checkpoints based on Validation Quadratic Weighted Kappa (QWK)
- Evaluates test set performance upon training completion
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from severity_model import ContextAwareSeverityNetwork
from severity_loss import CORALLoss, CORNLoss, FocalLoss, StandardCELoss, compute_severity_metrics


class FeatureDataset(Dataset):
    def __init__(self, data_dict: dict, mask: np.ndarray):
        self.vis_feats = data_dict['vis_feats'][mask]
        self.species_ids = data_dict['species_ids'][mask]
        self.grade_ctx = data_dict['grade_ctx'][mask]
        self.ood_ctx = data_dict['ood_ctx'][mask]
        self.unc_ctx = data_dict['unc_ctx'][mask]
        self.targets = data_dict['targets'][mask]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.vis_feats[idx],
            self.species_ids[idx],
            self.grade_ctx[idx],
            self.ood_ctx[idx],
            self.unc_ctx[idx],
            self.targets[idx]
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 4 Context-Aware Severity Network")
    parser.add_argument("--dataset_path", type=str, default="outputs/features/phase4_dataset.pt", help="Path to cached feature dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/phase4_severity", help="Output directory")
    parser.add_argument("--ablation", type=str, default="E", choices=["A", "B", "C", "D", "E"], help="Ablation mode")
    parser.add_argument("--loss", type=str, default="coral", choices=["coral", "corn", "ce", "focal"], help="Loss function")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for vis, spec, gr_ctx, ood_c, unc_c, targets in loader:
        vis = vis.to(device)
        spec = spec.to(device)
        gr_ctx = gr_ctx.to(device)
        ood_c = ood_c.to(device)
        unc_c = unc_c.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(vis, spec, gr_ctx, ood_c, unc_c)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(targets)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for vis, spec, gr_ctx, ood_c, unc_c, targets in loader:
        vis = vis.to(device)
        spec = spec.to(device)
        gr_ctx = gr_ctx.to(device)
        ood_c = ood_c.to(device)
        unc_c = unc_c.to(device)

        logits = model(vis, spec, gr_ctx, ood_c, unc_c)
        preds = model.predict_rank(logits).cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

    return compute_severity_metrics(all_targets, all_preds), np.array(all_targets), np.array(all_preds)


def run_training(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path missing: {args.dataset_path}")

    data_dict = torch.load(args.dataset_path, map_location="cpu")
    splits = data_dict['splits']

    train_mask = (splits == 'train')
    val_mask = (splits == 'val')
    test_mask = (splits == 'test')

    train_ds = FeatureDataset(data_dict, train_mask)
    val_ds = FeatureDataset(data_dict, val_mask)
    test_ds = FeatureDataset(data_dict, test_mask)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Compute Class Weights for Imbalance Handling
    train_targets = data_dict['targets'][train_mask].numpy()
    class_counts = np.bincount(train_targets, minlength=5)
    total_samples = len(train_targets)
    class_weights = total_samples / (5.0 * np.maximum(class_counts, 1))
    class_weights_t = torch.from_numpy(class_weights).float().to(device)

    # Initialize Model
    model = ContextAwareSeverityNetwork(
        vis_in_dim=768,
        num_species=5,
        num_severity_classes=5,
        head_type=args.loss,
        ablation_mode=args.ablation
    ).to(device)

    # Loss Selection
    if args.loss == "coral":
        criterion = CORALLoss(num_classes=5, class_weights=class_weights_t)
    elif args.loss == "corn":
        criterion = CORNLoss(num_classes=5)
    elif args.loss == "focal":
        criterion = FocalLoss(num_classes=5, gamma=2.0, alpha=class_weights_t)
    elif args.loss == "ce":
        criterion = StandardCELoss(num_classes=5, class_weights=class_weights_t)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_qwk = -1.0
    best_ckpt_path = os.path.join(args.output_dir, f"model_ablation_{args.ablation}_loss_{args.loss}_best.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        val_qwk = val_metrics["QWK"]

        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args)
            }, best_ckpt_path)

    # Load best checkpoint and evaluate test set
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_targets, test_preds = evaluate_model(model, test_loader, device)

    return test_metrics, best_val_qwk, best_ckpt_path


if __name__ == "__main__":
    args = parse_args()
    print(f"--- Training CASM | Ablation: {args.ablation} | Loss: {args.loss} ---")
    metrics, best_qwk, ckpt_path = run_training(args)
    print("\n--- TEST METRICS ---")
    print(json.dumps(metrics, indent=4))
