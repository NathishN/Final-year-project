"""
error_analysis.py
==================
Error Analysis and Visualization Module for ConvNeXt-Tiny Baseline.

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Functions:
- Identifies misclassified test instances
- Analyzes error distribution across domains (fruits360 vs real-world)
- Exports top confidence error samples to CSV
- Visualizes misclassified samples grid with ground truth, predicted species, confidence score, and source domain
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from fruit_quality_dataset import FruitQualityDataset
from model import build_convnext_model
from utils import setup_output_dirs, setup_logger, CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Error Analysis for ConvNeXt-Tiny Baseline")
    parser.add_argument("--csv_path", type=str, default="master_dataset.csv", help="Path to master_dataset.csv")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Outputs directory")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution")
    parser.add_argument("--top_k", type=int, default=16, help="Number of sample misclassifications to plot in grid")
    return parser.parse_args()


@torch.no_grad()
def collect_predictions_and_errors(model, dataloader, device):
    model.eval()
    results = []

    for imgs, labels, metas in tqdm(dataloader, desc="Analyzing Errors", leave=False):
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        confidences = torch.max(probs, dim=1).values.cpu().numpy()
        probs_np = probs.cpu().numpy()
        labels_np = labels.numpy()

        for i in range(len(labels_np)):
            true_idx = labels_np[i]
            pred_idx = preds[i]
            conf = float(confidences[i])
            img_path = metas["image_path"][i]
            domain = metas["domain"][i]
            fruit_name = metas["fruit"][i]

            results.append({
                "image_path": img_path,
                "domain": domain,
                "true_label": CLASS_NAMES[true_idx],
                "pred_label": CLASS_NAMES[pred_idx],
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "confidence": conf,
                "is_correct": bool(true_idx == pred_idx),
            })

    return pd.DataFrame(results)


def visualize_misclassified_grid(df_errors: pd.DataFrame, save_path: str, top_k: int = 16):
    """Plot a grid of high-confidence misclassifications."""
    if len(df_errors) == 0:
        print("[Error Analysis] No misclassified samples found!")
        return

    # Sort by highest confidence errors (most confident wrong predictions)
    df_sorted = df_errors.sort_values(by="confidence", ascending=False).head(top_k)
    num_samples = len(df_sorted)
    cols = 4
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), dpi=300)
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for idx in range(rows * cols):
        ax = axes[idx]
        if idx < num_samples:
            row = df_sorted.iloc[idx]
            img_path = row["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Image Read Error\n{e}", ha="center", va="center")

            title_str = (
                f"True: {row['true_label']}\n"
                f"Pred: {row['pred_label']} ({row['confidence']*100:.1f}%)\n"
                f"Domain: {row['domain']}"
            )
            ax.set_title(title_str, fontsize=9, color="red", fontweight="bold")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle("ConvNeXt-Tiny Top High-Confidence Misclassifications", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Error Analysis] Grid visualization saved to: {save_path}")


def main():
    args = parse_args()
    output_paths = setup_output_dirs(args.base_dir)
    logger = setup_logger(os.path.join(output_paths["logs"], "error_analysis.log"))

    logger.info("==================================================================")
    logger.info("Starting ConvNeXt-Tiny Baseline Error Analysis")
    logger.info("==================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint missing: {args.checkpoint}")

    model = build_convnext_model(num_classes=5, pretrained=False, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from: {args.checkpoint}")

    test_dataset = FruitQualityDataset(
        csv_path=args.csv_path,
        split="test",
        label_type="fruit",
        img_size=args.img_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    df_results = collect_predictions_and_errors(model, test_loader, device)

    total_test = len(df_results)
    total_correct = df_results["is_correct"].sum()
    total_errors = total_test - total_correct
    accuracy = total_correct / total_test * 100.0

    logger.info("\n" + "=" * 60)
    logger.info("ERROR ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Test Samples   : {total_test:,}")
    logger.info(f"Correct Predictions  : {total_correct:,} ({accuracy:.2f}%)")
    logger.info(f"Misclassifications   : {total_errors:,} ({100.0 - accuracy:.2f}%)")

    # Error distribution by domain
    domain_errors = df_results.groupby("domain")["is_correct"].agg(
        total="count",
        errors=lambda x: (x == False).sum(),
        accuracy=lambda x: x.mean() * 100
    )
    logger.info("\nDomain-wise Error Breakdown:\n" + str(domain_errors))

    # Error distribution by class
    class_errors = df_results.groupby("true_label")["is_correct"].agg(
        total="count",
        errors=lambda x: (x == False).sum(),
        accuracy=lambda x: x.mean() * 100
    )
    logger.info("\nPer-Class Error Breakdown:\n" + str(class_errors))

    # Top Confusion Pairs
    df_errors = df_results[df_results["is_correct"] == False].copy()
    if len(df_errors) > 0:
        df_errors["confusion_pair"] = df_errors["true_label"] + " -> " + df_errors["pred_label"]
        top_pairs = df_errors["confusion_pair"].value_counts().head(5)
        logger.info("\nTop 5 Misclassification Confusion Pairs:\n" + str(top_pairs))

    # Export errors CSV to outputs/misclassified/misclassifications.csv
    errors_csv_path = os.path.join(output_paths["misclassified"], "misclassifications.csv")
    df_errors.to_csv(errors_csv_path, index=False)
    logger.info(f"Exported misclassified samples CSV to: {errors_csv_path}")

    # Visualize misclassified samples grid to outputs/misclassified/misclassified_samples.png
    grid_plot_path = os.path.join(output_paths["misclassified"], "misclassified_samples.png")
    visualize_misclassified_grid(df_errors, grid_plot_path, top_k=args.top_k)

    logger.info("Error analysis complete!")


if __name__ == "__main__":
    main()
