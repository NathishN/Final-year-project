"""
train_baseline.py
=================
Complete Research Training Pipeline for ConvNeXt-Tiny Baseline Fruit Classifier.

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Research-Grade Enhancements:
- Backbone: ConvNeXt-Tiny (ImageNet pretrained via timm)
- Imbalance Handling: Class-Balanced Focal Loss (CB-Focal Loss, beta=0.9999, gamma=2.0)
- Domain Shift Mitigation: WeightedRandomSampler based on (fruit x domain) stratums
- Label Smoothing: epsilon = 0.1 (prevents overconfidence before Phase 5 Calibration)
- Optimization: AdamW (lr=3e-4, weight_decay=1e-2)
- LR Scheduler: CosineAnnealingLR (T_max=30, eta_min=1e-6)
- Mixed Precision: PyTorch AMP (autocast + GradScaler)
- Gradient Clipping: max_norm = 1.0
- Early Stopping: Patience = 5 epochs (monitored on Validation Macro-F1)
- Hardware Target: RTX 3050 6GB GPU / 16GB RAM
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fruit_quality_dataset import FruitQualityDataset
from model import build_convnext_model
from loss import get_loss_function
from utils import (
    set_seed,
    setup_output_dirs,
    setup_logger,
    calculate_metrics,
    plot_training_curves,
    CheckpointSaver,
    CLASS_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny Baseline Fruit Classifier")
    parser.add_argument("--csv_path", type=str, default="master_dataset.csv", help="Path to master_dataset.csv")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Directory to save experiment outputs")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay coefficient")
    parser.add_argument("--loss_type", type=str, default="cb_focal", choices=["cb_focal", "focal", "weighted_ce", "ce"], help="Loss function choice")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing parameter")
    parser.add_argument("--use_sampler", action="store_true", default=True, help="Use WeightedRandomSampler for domain imbalance")
    parser.add_argument("--img_size", type=int, default=224, help="Image input resolution")
    parser.add_argument("--num_workers", type=int, default=4, help="PyTorch DataLoader worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for imgs, labels, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    return epoch_loss, metrics


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for imgs, labels, _ in tqdm(dataloader, desc="Validation", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    return epoch_loss, metrics


def main():
    args = parse_args()

    # 1. Reproducibility & Output Setup
    set_seed(args.seed)
    output_paths = setup_output_dirs(args.base_dir)
    log_file = os.path.join(output_paths["logs"], "training.log")
    logger = setup_logger(log_file)

    logger.info("==================================================================")
    logger.info("Starting Research ConvNeXt-Tiny Baseline Fruit Classifier Training")
    logger.info("==================================================================")
    logger.info(f"Arguments: {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Computation Device: CUDA ({torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.2f} GB)")
    else:
        logger.info("Computation Device: CPU")

    # 2. Datasets & Loaders with Domain-Balanced Sampling
    logger.info(f"Loading datasets from {args.csv_path}...")
    train_dataset = FruitQualityDataset(
        csv_path=args.csv_path,
        split="train",
        label_type="fruit",
        img_size=args.img_size,
    )
    val_dataset = FruitQualityDataset(
        csv_path=args.csv_path,
        split="val",
        label_type="fruit",
        img_size=args.img_size,
    )

    if args.use_sampler:
        logger.info("[Domain Shift Mitigation] Building WeightedRandomSampler based on (fruit x domain) stratums...")
        sampler = train_dataset.get_domain_balanced_sampler()
        shuffle_train = False
    else:
        sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Class Counts
    fruit_counts = train_dataset.df["fruit"].value_counts()
    class_counts = [int(fruit_counts.get(name, 0)) for name in CLASS_NAMES]
    logger.info(f"Train Class Counts (Apple, Banana, Mango, Orange, Tomato): {class_counts}")

    # 3. Model, Loss, Optimizer & Scheduler
    logger.info("Building ConvNeXt-Tiny pretrained backbone...")
    model = build_convnext_model(num_classes=5, pretrained=True, device=device)

    criterion = get_loss_function(
        loss_name=args.loss_type,
        class_counts=class_counts,
        label_smoothing=args.label_smoothing,
        device=device
    )
    logger.info(f"Selected Loss Function: {args.loss_type.upper()} (Label Smoothing eps={args.label_smoothing})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    checkpoint_file = os.path.join(output_paths["checkpoints"], "best_model.pth")
    saver = CheckpointSaver(checkpoint_file, logger=logger)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    patience_counter = 0

    # 4. Training Loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} [LR: {optimizer.param_groups[0]['lr']:.6f}] ---")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["macro_f1"])
        history["val_f1"].append(val_metrics["macro_f1"])

        logger.info(
            f"Epoch {epoch:02d} Summary | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['macro_f1']:.4f}"
        )
        logger.info(
            f"Epoch {epoch:02d} Summary | "
            f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.4f} | Val F1:   {val_metrics['macro_f1']:.4f}"
        )

        # Checkpoint & Early Stopping
        is_best = saver.step(val_metrics["macro_f1"], model, optimizer, epoch, val_metrics)
        if is_best:
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement in Val Macro-F1. Early stopping counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info(f"\n[Early Stopping] Triggered after {epoch} epochs without validation macro-F1 improvement.")
            break

    elapsed = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed / 60:.2f} minutes.")

    # 5. Save Metrics History & Plots
    history_file = os.path.join(output_paths["metrics"], "history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

    curves_plot_file = os.path.join(output_paths["plots"], "training_curves.png")
    plot_training_curves(history, curves_plot_file)

    logger.info(f"Baseline training successfully completed! Best model saved at: {checkpoint_file}")


if __name__ == "__main__":
    main()
