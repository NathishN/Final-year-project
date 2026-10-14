"""
evaluate_baseline.py
====================
Evaluation Pipeline for ConvNeXt-Tiny Baseline Fruit Classifier.

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Outputs:
- outputs/metrics/classification_report.txt
- outputs/metrics/test_metrics.json
- outputs/metrics/domain_metrics.json
- outputs/metrics/per_class_accuracy.csv
- outputs/confusion_matrix/confusion_matrix.png
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from fruit_quality_dataset import FruitQualityDataset
from model import build_convnext_model
from utils import (
    setup_output_dirs,
    setup_logger,
    calculate_metrics,
    plot_confusion_matrix,
    CLASS_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt-Tiny Baseline Model on Test Split")
    parser.add_argument("--csv_path", type=str, default="master_dataset.csv", help="Path to master_dataset.csv")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Outputs directory")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_domains = []

    for imgs, labels, metas in tqdm(dataloader, desc="Testing Model", leave=False):
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_domains.extend(metas["domain"])

    return np.array(all_targets), np.array(all_preds), np.array(all_probs), np.array(all_domains)


def main():
    args = parse_args()

    # 1. Output Paths & Logging
    output_paths = setup_output_dirs(args.base_dir)
    logger = setup_logger(os.path.join(output_paths["logs"], "evaluation.log"))

    logger.info("==================================================================")
    logger.info("Evaluating ConvNeXt-Tiny Baseline Fruit Classifier on Test Split")
    logger.info("==================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target Device: {device}")

    # 2. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}. Run train_baseline.py first.")

    logger.info(f"Loading best checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = build_convnext_model(num_classes=5, pretrained=False, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint (Epoch: {checkpoint.get('epoch', '?')}, Val Macro-F1: {checkpoint.get('best_val_f1', 0.0):.4f})")

    # 3. Test Dataset & DataLoader
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
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # 4. Run Evaluation
    y_true, y_pred, y_prob, y_domains = evaluate_model(model, test_loader, device)

    # 5. Compute Metrics & Reports
    metrics = calculate_metrics(y_true, y_pred, class_names=CLASS_NAMES)
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    logger.info("\n" + "=" * 60)
    logger.info("OVERALL TEST METRICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy        : {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Test Macro Precision : {metrics['macro_precision'] * 100:.2f}%")
    logger.info(f"Test Macro Recall    : {metrics['macro_recall'] * 100:.2f}%")
    logger.info(f"Test Macro-F1        : {metrics['macro_f1'] * 100:.2f}%")
    logger.info(f"Test Weighted-F1     : {metrics['weighted_f1'] * 100:.2f}%")
    logger.info("\n" + "=" * 60)
    logger.info("PER-CLASS CLASSIFICATION REPORT")
    logger.info("=" * 60)
    logger.info("\n" + report_str)

    # Save Classification Report Text
    report_file = os.path.join(output_paths["metrics"], "classification_report.txt")
    with open(report_file, "w") as f:
        f.write("CONVNEXT-TINY BASELINE TEST EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(report_str)
    logger.info(f"Saved text report to: {report_file}")

    # Save JSON metrics
    metrics_file = os.path.join(output_paths["metrics"], "test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved JSON metrics to: {metrics_file}")

    # 6. Per-Domain Metrics
    domain_metrics = {}
    unique_domains = np.unique(y_domains)
    for dom in unique_domains:
        mask = (y_domains == dom)
        if np.sum(mask) > 0:
            dom_acc = accuracy_score(y_true[mask], y_pred[mask])
            _, _, dom_f1, _ = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="macro", zero_division=0
            )
            domain_metrics[dom] = {
                "sample_count": int(np.sum(mask)),
                "accuracy": float(dom_acc),
                "macro_f1": float(dom_f1),
            }

    domain_file = os.path.join(output_paths["metrics"], "domain_metrics.json")
    with open(domain_file, "w") as f:
        json.dump(domain_metrics, f, indent=4)
    logger.info(f"Saved domain metrics to: {domain_file}")

    # 7. Per-Class Accuracy CSV
    per_class_rows = []
    for i, name in enumerate(CLASS_NAMES):
        mask = (y_true == i)
        if np.sum(mask) > 0:
            cls_acc = accuracy_score(y_true[mask], y_pred[mask])
            per_class_rows.append({
                "class_idx": i,
                "fruit": name,
                "total_samples": int(np.sum(mask)),
                "correct": int(np.sum(y_true[mask] == y_pred[mask])),
                "accuracy": float(cls_acc),
            })
    df_per_class = pd.DataFrame(per_class_rows)
    per_class_file = os.path.join(output_paths["metrics"], "per_class_accuracy.csv")
    df_per_class.to_csv(per_class_file, index=False)
    logger.info(f"Saved per-class accuracy CSV to: {per_class_file}")

    # 8. Plot Confusion Matrix
    cm_path = os.path.join(output_paths["confusion_matrix"], "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path, class_names=CLASS_NAMES)

    logger.info("Evaluation pipeline execution complete!")


if __name__ == "__main__":
    main()
