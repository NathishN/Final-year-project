"""
evaluate_phase4_severity.py
============================
Evaluation Pipeline for Phase 4 Context-Aware Severity Modeling (CASM).

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Outputs:
- Comprehensive per-domain evaluation (Fruits360, real_grade, real_quality, real_severity)
- Per-species breakdown (Apple, Banana, Mango, Orange, Tomato)
- Error matrix and classification report
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader

from severity_model import ContextAwareSeverityNetwork
from severity_loss import compute_severity_metrics


class FullFeatureDataset(Dataset):
    def __init__(self, data_dict: dict, mask: np.ndarray):
        self.vis_feats = data_dict['vis_feats'][mask]
        self.species_ids = data_dict['species_ids'][mask]
        self.grade_ctx = data_dict['grade_ctx'][mask]
        self.ood_ctx = data_dict['ood_ctx'][mask]
        self.unc_ctx = data_dict['unc_ctx'][mask]
        self.targets = data_dict['targets'][mask]
        self.domains = data_dict['domains'][mask]
        self.fruits = data_dict['fruits'][mask]
        self.paths = data_dict['paths'][mask]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.vis_feats[idx],
            self.species_ids[idx],
            self.grade_ctx[idx],
            self.ood_ctx[idx],
            self.unc_ctx[idx],
            self.targets[idx],
            self.domains[idx],
            self.fruits[idx],
            self.paths[idx]
        )


def evaluate_checkpoint(checkpoint_path: str, dataset_path: str = "outputs/features/phase4_dataset.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]

    data_dict = torch.load(dataset_path, map_location="cpu")
    splits = data_dict['splits']
    test_mask = (splits == 'test')

    test_ds = FullFeatureDataset(data_dict, test_mask)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = ContextAwareSeverityNetwork(
        vis_in_dim=768,
        num_species=5,
        num_severity_classes=5,
        head_type=args["loss"],
        ablation_mode=args["ablation"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    all_domains = []
    all_fruits = []
    all_paths = []

    with torch.no_grad():
        for vis, spec, gr_ctx, ood_c, unc_c, targets, doms, frs, paths in loader:
            vis = vis.to(device)
            spec = spec.to(device)
            gr_ctx = gr_ctx.to(device)
            ood_c = ood_c.to(device)
            unc_c = unc_c.to(device)

            logits = model(vis, spec, gr_ctx, ood_c, unc_c)
            preds = model.predict_rank(logits).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_domains.extend(doms)
            all_fruits.extend(frs)
            all_paths.extend(paths)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_domains = np.array(all_domains)
    all_fruits = np.array(all_fruits)

    # 1. Overall Metrics
    overall_metrics = compute_severity_metrics(all_targets, all_preds)

    # 2. Per-Domain Breakdown
    domain_metrics = {}
    for dom in np.unique(all_domains):
        dom_mask = (all_domains == dom)
        if np.sum(dom_mask) > 0:
            domain_metrics[dom] = compute_severity_metrics(all_targets[dom_mask], all_preds[dom_mask])

    # 3. Per-Species Breakdown
    species_metrics = {}
    for fr in np.unique(all_fruits):
        fr_mask = (all_fruits == fr)
        if np.sum(fr_mask) > 0:
            species_metrics[fr] = compute_severity_metrics(all_targets[fr_mask], all_preds[fr_mask])

    results = {
        "overall": overall_metrics,
        "by_domain": domain_metrics,
        "by_species": species_metrics,
        "confusion_matrix": confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4]).tolist()
    }

    return results, all_targets, all_preds, all_domains, all_fruits, all_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="outputs/features/phase4_dataset.pt")
    args = parser.parse_args()

    results, targets, preds, doms, frs, paths = evaluate_checkpoint(args.checkpoint, args.dataset_path)
    print("\n" + "=" * 60)
    print("PHASE 4 CASM EVALUATION REPORT")
    print("=" * 60)
    print("Overall Test Metrics:")
    print(json.dumps(results["overall"], indent=4))
