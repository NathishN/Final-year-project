"""
utils.py
========
Utility functions for reproducibility, metric calculation, logging, plotting, and checkpointing.

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"
"""

import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import torch


CLASS_NAMES = ["Apple", "Banana", "Mango", "Orange", "Tomato"]


def set_seed(seed: int = 42):
    """Ensure strict research reproducibility across PyTorch, NumPy, and Python built-in random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Reproducibility] Random seed set to: {seed}")


def setup_output_dirs(base_dir: str = "outputs"):
    """Create directory structure required for logs, checkpoints, metrics, plots, and confusion matrix."""
    subdirs = ["checkpoints", "logs", "metrics", "plots", "confusion_matrix", "misclassified"]
    paths = {}
    for d in subdirs:
        path = os.path.join(base_dir, d)
        os.makedirs(path, exist_ok=True)
        paths[d] = path
    print(f"[Outputs] Output directories initialized under: {os.path.abspath(base_dir)}")
    return paths


def setup_logger(log_file_path: str):
    """Configures double logging to both stdout terminal and log text file."""
    logger = logging.getLogger("BaselineLogger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear existing handlers

    formatter = logging.formatters = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names=CLASS_NAMES) -> dict:
    """
    Computes comprehensive classification metrics:
    - Overall Accuracy
    - Macro Precision, Recall, F1
    - Weighted Precision, Recall, F1
    - Per-class metrics
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class breakdown
    prec_class, rec_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(prec_class[i]),
            "recall": float(rec_class[i]),
            "f1_score": float(f1_class[i]),
            "support": int(support_class[i]),
        }

    return {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class_metrics,
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, class_names=CLASS_NAMES):
    """Generates and saves publication-quality normalized confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Normalized Accuracy"},
    )
    ax.set_title("ConvNeXt-Tiny Species Classification Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved to: {save_path}")


def plot_training_curves(history: dict, save_path: str):
    """Generates loss and macro-F1 training progress curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-s", label="Val Loss", linewidth=2)
    ax1.set_title("Loss Trajectory", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Macro-F1 Curve
    ax2.plot(epochs, history["train_f1"], "b-o", label="Train Macro-F1", linewidth=2)
    ax2.plot(epochs, history["val_f1"], "g-^", label="Val Macro-F1", linewidth=2)
    ax2.set_title("Validation Macro-F1 Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Macro-F1", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved to: {save_path}")


class CheckpointSaver:
    """Saves best model checkpoint based on Validation Macro-F1 metric."""

    def __init__(self, checkpoint_path: str, logger=None):
        self.checkpoint_path = checkpoint_path
        self.logger = logger
        self.best_f1 = -1.0

    def step(self, val_f1: float, model: torch.nn.Module, optimizer, epoch: int, metrics: dict):
        if val_f1 > self.best_f1:
            msg = f"Validation Macro-F1 improved from {self.best_f1:.4f} to {val_f1:.4f}. Saving best model checkpoint."
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            self.best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": val_f1,
                    "metrics": metrics,
                },
                self.checkpoint_path,
            )
            return True
        return False
