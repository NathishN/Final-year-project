"""
evaluate_phase3_coral.py
========================
Phase 3 Evaluation Pipeline: Rank-Consistent Ordinal Regression (CORAL).

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Outputs:
- outputs/metrics/coral_test_metrics.json
- outputs/confusion_matrix/coral_ordinal_cm.png
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_recall_fscore_support, classification_report

import torch
from torch.utils.data import DataLoader

from fruit_quality_dataset import FruitQualityDataset
from model import build_coral_model
from utils import setup_output_dirs, setup_logger, plot_confusion_matrix


ORDINAL_CLASSES = ["1st_grade", "2nd_grade", "3rd_grade"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt-Tiny CORAL Ordinal Grader")
    parser.add_argument("--csv_path", type=str, default="master_dataset.csv", help="Path to master_dataset.csv")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Outputs base directory")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/coral_best_model.pth", help="Checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution")
    return parser.parse_args()


@torch.no_grad()
def evaluate_coral_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []

    for imgs, labels, _ in tqdm(dataloader, desc="Evaluating CORAL", leave=False):
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(imgs)

        preds = model.predict_rank(logits).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())
        all_logits.extend(logits.cpu().numpy())

    return np.array(all_targets), np.array(all_preds), np.array(all_logits)


def main():
    args = parse_args()
    output_paths = setup_output_dirs(args.base_dir)
    logger = setup_logger(os.path.join(output_paths["logs"], "coral_evaluation.log"))

    logger.info("==================================================================")
    logger.info("Evaluating Phase 3 CORAL Ordinal Quality Grader on Test Split")
    logger.info("==================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target Device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"CORAL checkpoint missing: {args.checkpoint}")

    model = build_coral_model(num_classes=3, pretrained=False, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded CORAL model from: {args.checkpoint} (Best Val QWK: {checkpoint.get('best_val_qwk', 0.0):.4f})")

    # Load test split for real_grade
    fruits_for_grade = ["Apple", "Mango", "Tomato"]
    test_ds = FruitQualityDataset(
        csv_path=args.csv_path,
        split="test",
        label_type="grade",
        domain="real_grade",
        fruits=fruits_for_grade,
        img_size=args.img_size,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

    y_true, y_pred, y_logits = evaluate_coral_model(model, test_loader, device)

    # Compute Ordinal Metrics
    mae = float(np.mean(np.abs(y_true - y_pred)))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    # Off-by-one error (percentage of predictions where |y_true - y_pred| <= 1)
    off_by_one = float(np.mean(np.abs(y_true - y_pred) <= 1) * 100.0)

    metrics = {
        "MAE": mae,
        "QWK": qwk,
        "Accuracy": float(acc),
        "Macro_Precision": float(prec),
        "Macro_Recall": float(rec),
        "Macro_F1": float(f1),
        "Off_By_One_Accuracy": off_by_one,
        "Test_Samples": len(y_true),
    }

    report_str = classification_report(y_true, y_pred, target_names=ORDINAL_CLASSES, digits=4)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 CORAL ORDINAL REGRESSION TEST METRICS")
    logger.info("=" * 60)
    logger.info(f"Mean Absolute Error (MAE)  : {mae:.4f}")
    logger.info(f"Quadratic Weighted Kappa   : {qwk:.4f}")
    logger.info(f"Accuracy                   : {acc * 100:.2f}%")
    logger.info(f"Off-By-One Tolerance Acc   : {off_by_one:.2f}%")
    logger.info(f"Macro-F1                   : {f1 * 100:.2f}%")
    logger.info("\n" + "=" * 60)
    logger.info("PER-RANK CLASSIFICATION REPORT")
    logger.info("=" * 60)
    logger.info("\n" + report_str)

    # Save metrics JSON
    metrics_file = os.path.join(output_paths["metrics"], "coral_test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved CORAL metrics to: {metrics_file}")

    # Plot Confusion Matrix
    cm_path = os.path.join(output_paths["confusion_matrix"], "coral_ordinal_cm.png")
    plot_confusion_matrix(y_true, y_pred, cm_path, class_names=ORDINAL_CLASSES)

    logger.info("CORAL evaluation complete!")


if __name__ == "__main__":
    main()
