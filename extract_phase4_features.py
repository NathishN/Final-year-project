"""
extract_phase4_features.py
===========================
Feature and Context Extractor for Phase 4 Context-Aware Severity Modeling (CASM).

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Outputs:
- outputs/features/phase4_dataset.pt (Contains vis_feats, species_ids, grade_ctx, ood_ctx, unc_ctx, targets, splits, domains, fruits, paths)
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import build_convnext_model, build_coral_model
from open_set_recognition import MahalanobisOOD


SPECIES_MAP = {
    "Apple": 0,
    "Banana": 1,
    "Mango": 2,
    "Orange": 3,
    "Tomato": 4
}


def map_severity(row):
    dom = row['domain']
    if dom == 'real_severity':
        s = str(row['severity_label_normalized']).lower()
        if s == 'fresh': return 0
        elif s == 'mild': return 1
        elif s == 'rotten': return 4
    elif dom == 'real_quality':
        q = str(row['quality_label_normalized']).lower()
        if q == 'good': return 0
        elif q == 'mixed': return 2
        elif q == 'bad': return 3
    elif dom == 'real_grade':
        g = str(row['grade_label_normalized']).lower()
        if g == 'grade_1': return 0
        elif g == 'grade_2': return 1
        elif g == 'grade_3': return 3
    elif dom == 'fruits360':
        return 0
    return -1


class ImageExtractDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback for broken file read
            image = Image.new('RGB', (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        species_id = SPECIES_MAP.get(row['fruit'], 0)
        target = map_severity(row)

        return image, species_id, target, idx


@torch.no_grad()
def main():
    csv_path = "master_dataset.csv"
    output_dir = "outputs/features"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    valid_df = df[df['split'] != 'EXCLUDED'].copy().reset_index(drop=True)

    print(f"Total valid samples to extract: {len(valid_df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extraction target device: {device}")

    # Image Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = ImageExtractDataset(valid_df, transform=transform)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    # Load Phase 1 Species Classifier Model
    p1_ckpt = "outputs/checkpoints/best_model.pth"
    model_p1 = build_convnext_model(num_classes=5, pretrained=False, device=device)
    if os.path.exists(p1_ckpt):
        ckpt1 = torch.load(p1_ckpt, map_location=device, weights_only=False)
        model_p1.load_state_dict(ckpt1['model_state_dict'])
        print(f"Loaded Phase 1 model checkpoint from {p1_ckpt}")
    model_p1.eval()

    # Load Phase 3 CORAL Grade Model
    p3_ckpt = "outputs/coral/checkpoints/coral_best_model.pth"
    model_p3 = build_coral_model(num_classes=3, pretrained=False, device=device)
    if os.path.exists(p3_ckpt):
        ckpt3 = torch.load(p3_ckpt, map_location=device, weights_only=False)
        model_p3.load_state_dict(ckpt3['state_dict'])
        print(f"Loaded Phase 3 CORAL model checkpoint from {p3_ckpt}")
    model_p3.eval()

    print("Step 1: Extracting visual embeddings, species logits, and grade logits...")
    vis_feats_list = []
    p1_logits_list = []
    p3_logits_list = []
    species_ids_list = []
    targets_list = []

    total_batches = len(loader)
    for b_idx, (imgs, spec_ids, targets, idxs) in enumerate(loader):
        if b_idx % 20 == 0:
            print(f"  Batch {b_idx}/{total_batches} ({b_idx * 256}/{len(valid_df)} samples)...", flush=True)
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            # Bottleneck visual features (768-D)
            emb = model_p1.extract_features(imgs)
            if len(emb.shape) == 4:
                emb = torch.mean(emb, dim=[2, 3])  # Global Avg Pooling

            # Phase 1 Logits (5-class species)
            p1_logits = model_p1(imgs)

            # Phase 3 CORAL Grade Logits (2-thresholds)
            p3_logits = model_p3(imgs)

        vis_feats_list.append(emb.cpu().numpy())
        p1_logits_list.append(p1_logits.cpu().numpy())
        p3_logits_list.append(p3_logits.cpu().numpy())
        species_ids_list.append(spec_ids.numpy())
        targets_list.append(targets.numpy())

    vis_feats = np.concatenate(vis_feats_list, axis=0)      # (N, 768)
    p1_logits = np.concatenate(p1_logits_list, axis=0)      # (N, 5)
    p3_logits = np.concatenate(p3_logits_list, axis=0)      # (N, 2)
    species_ids = np.concatenate(species_ids_list, axis=0)  # (N,)
    targets = np.concatenate(targets_list, axis=0)          # (N,)

    # Step 2: Fit Mahalanobis OOD Detector on Training Split Features
    print("Step 2: Fitting Mahalanobis OOD detector on training split features...")
    train_mask = (valid_df['split'] == 'train').values
    train_feats = vis_feats[train_mask]
    train_species = species_ids[train_mask]

    mah_detector = MahalanobisOOD(num_classes=5)
    mah_detector.fit(train_feats, train_species)

    # Compute Mahalanobis scores for all samples
    mah_scores = mah_detector.score(vis_feats)  # (N,) distance score

    # Step 3: Compute Context Vectors
    print("Step 3: Constructing Grade, OOD, and Uncertainty context vectors...")
    # A. Uncertainty Context
    p1_probs = np.exp(p1_logits - np.max(p1_logits, axis=1, keepdims=True))
    p1_probs /= np.sum(p1_probs, axis=1, keepdims=True)

    max_conf = np.max(p1_probs, axis=1)  # (N,)
    entropy = -np.sum(p1_probs * np.log(p1_probs + 1e-8), axis=1)  # (N,)
    unc_ctx = np.stack([entropy, max_conf], axis=1).astype(np.float32)  # (N, 2)

    # B. OOD Context
    p_ood = 1.0 - max_conf
    # Normalize mah scores for numerical stability
    mah_norm = (mah_scores - np.mean(mah_scores)) / (np.std(mah_scores) + 1e-8)
    ood_ctx = np.stack([mah_norm, p_ood], axis=1).astype(np.float32)  # (N, 2)

    # C. Grade Context from Phase 3 CORAL
    # p3_logits are CORAL binary logits for P(grade > 0) and P(grade > 1)
    sig_g0 = 1.0 / (1.0 + np.exp(-p3_logits[:, 0]))
    sig_g1 = 1.0 / (1.0 + np.exp(-p3_logits[:, 1]))

    p_g1 = 1.0 - sig_g0
    p_g2 = np.clip(sig_g0 - sig_g1, 0.0, 1.0)
    p_g3 = sig_g1

    grade_probs = np.stack([p_g1, p_g2, p_g3], axis=1)  # (N, 3)
    # Re-normalize grade_probs
    grade_probs /= np.sum(grade_probs, axis=1, keepdims=True) + 1e-8
    max_g_conf = np.max(grade_probs, axis=1, keepdims=True)

    grade_ctx = np.hstack([grade_probs, max_g_conf]).astype(np.float32)  # (N, 4)

    # Step 4: Package and Save to PyTorch File
    dataset_dict = {
        "vis_feats": torch.from_numpy(vis_feats).float(),
        "species_ids": torch.from_numpy(species_ids).long(),
        "grade_ctx": torch.from_numpy(grade_ctx).float(),
        "ood_ctx": torch.from_numpy(ood_ctx).float(),
        "unc_ctx": torch.from_numpy(unc_ctx).float(),
        "targets": torch.from_numpy(targets).long(),
        "splits": valid_df['split'].values,
        "domains": valid_df['domain'].values,
        "fruits": valid_df['fruit'].values,
        "paths": valid_df['image_path'].values
    }

    save_path = os.path.join(output_dir, "phase4_dataset.pt")
    torch.save(dataset_dict, save_path)
    print(f"\nFeature extraction complete! Saved dataset to {save_path}")
    print(f"Visual Features shape : {vis_feats.shape}")
    print(f"Grade Context shape   : {grade_ctx.shape}")
    print(f"OOD Context shape     : {ood_ctx.shape}")
    print(f"Uncertainty shape     : {unc_ctx.shape}")
    print(f"Targets shape         : {targets.shape}")


if __name__ == "__main__":
    main()
