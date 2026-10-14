"""
export_features.py
==================
Feature and Logit Extractor for Phase 2 (Energy OOD) and Phase 3 (CORAL Ordinal).

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Outputs:
- outputs/features/train_features.npy  (N_train, 768)
- outputs/features/train_logits.npy    (N_train, 5)
- outputs/features/train_probs.npy     (N_train, 5)
- outputs/features/train_labels.npy    (N_train,)
- outputs/features/train_domains.npy   (N_train,)
- outputs/features/test_features.npy   (N_test, 768)
- outputs/features/test_logits.npy     (N_test, 5)
- outputs/features/test_probs.npy      (N_test, 5)
- outputs/features/test_labels.npy     (N_test,)
- outputs/features/test_domains.npy    (N_test,)
- outputs/features/feature_metadata.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from fruit_quality_dataset import FruitQualityDataset
from model import build_convnext_model
from utils import setup_output_dirs, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Export Bottleneck Features and Logits")
    parser.add_argument("--csv_path", type=str, default="master_dataset.csv", help="Path to master_dataset.csv")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Outputs base directory")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution")
    return parser.parse_args()


@torch.no_grad()
def extract_dataset_features(model, dataloader, device):
    model.eval()
    features_list = []
    logits_list = []
    probs_list = []
    labels_list = []
    domains_list = []
    paths_list = []

    for imgs, labels, metas in tqdm(dataloader, desc="Extracting Features", leave=False):
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            # Extract 768-dim bottleneck embeddings prior to classifier linear head
            emb = model.extract_features(imgs)  # shape: (B, 768, 7, 7) or (B, 768)
            if len(emb.shape) == 4:
                emb = torch.mean(emb, dim=[2, 3])  # Global Avg Pooling

            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)

        features_list.append(emb.cpu().numpy())
        logits_list.append(logits.cpu().numpy())
        probs_list.append(probs.cpu().numpy())
        labels_list.extend(labels.numpy())
        domains_list.extend(metas["domain"])
        paths_list.extend(metas["image_path"])

    features_arr = np.concatenate(features_list, axis=0)
    logits_arr = np.concatenate(logits_list, axis=0)
    probs_arr = np.concatenate(probs_list, axis=0)
    labels_arr = np.array(labels_list, dtype=np.int64)
    domains_arr = np.array(domains_list, dtype=object)

    return features_arr, logits_arr, probs_arr, labels_arr, domains_arr, paths_list


def main():
    args = parse_args()
    
    # Feature directory under outputs/features/
    feature_dir = os.path.join(args.base_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)

    logger = setup_logger(os.path.join(args.base_dir, "logs", "feature_export.log"))

    logger.info("==================================================================")
    logger.info("Starting ConvNeXt-Tiny Feature and Logit Extraction")
    logger.info("==================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    model = build_convnext_model(num_classes=5, pretrained=False, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Successfully loaded model checkpoint from: {args.checkpoint}")

    # Process both Train and Test splits for downstream OOD and Calibration
    splits = ["train", "test"]
    metadata_summary = {}

    for split in splits:
        logger.info(f"\nProcessing '{split}' split...")
        ds = FruitQualityDataset(
            csv_path=args.csv_path,
            split=split,
            label_type="fruit",
            img_size=args.img_size,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device == "cuda"),
        )

        feats, logits, probs, labels, domains, paths = extract_dataset_features(model, loader, device)

        np.save(os.path.join(feature_dir, f"{split}_features.npy"), feats)
        np.save(os.path.join(feature_dir, f"{split}_logits.npy"), logits)
        np.save(os.path.join(feature_dir, f"{split}_probs.npy"), probs)
        np.save(os.path.join(feature_dir, f"{split}_labels.npy"), labels)
        np.save(os.path.join(feature_dir, f"{split}_domains.npy"), domains)

        metadata_summary[split] = {
            "num_samples": len(labels),
            "feature_dim": feats.shape[1],
            "num_classes": logits.shape[1],
            "domains": list(np.unique(domains)),
        }
        logger.info(f"Saved {split} features shape: {feats.shape}, logits shape: {logits.shape}")

    # Save feature metadata json
    meta_path = os.path.join(feature_dir, "feature_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata_summary, f, indent=4)

    logger.info(f"\nFeature extraction complete! All arrays saved in: {os.path.abspath(feature_dir)}")


if __name__ == "__main__":
    main()
