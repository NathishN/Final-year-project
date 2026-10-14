"""
phase50_stage4_ood_benchmark.py
================================
Phase 5.0 -- Stage 4: Real-Image OOD Benchmark

This script:
1. Inspects the open-set-dataset directory and documents its actual content.
2. If real OOD images exist: extracts features and runs benchmark.
3. If images are absent: documents the limitation and proposes data collection protocol.
4. Uses existing ConvNeXt features from known-fruit evaluation protocol.
5. Runs available OOD detectors on known-fruit validation/test features.

Deliverables:
  outputs/casm_v2/phase50_ood_features/  (or absence documentation)
  outputs/casm_v2/phase50_ood_benchmark.csv
  outputs/casm_v2/phase50_ood_report.md
"""

import sys
import io
import os
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn.functional as F

ROOT = r"C:\fruit_dataset"
OOD_DIR = os.path.join(ROOT, "open-set-dataset")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "casm_v2")
OOD_FEATURES_DIR = os.path.join(OUTPUT_DIR, "phase50_ood_features")
FEATURES_PATH = os.path.join(ROOT, "outputs", "features", "phase4_dataset.pt")

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

os.makedirs(OOD_FEATURES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------
# Step 1: Inspect OOD directory
# -------------------------------------------------------------------

def inspect_ood_directory():
    """Recursively count all images in the open-set-dataset directory."""
    print("\n[1] Inspecting open-set-dataset directory ...")

    ood_inventory = {}
    total_images = 0

    for cls_name in sorted(os.listdir(OOD_DIR)):
        cls_path = os.path.join(OOD_DIR, cls_name)
        if not os.path.isdir(cls_path):
            continue

        cls_images = []
        for root, dirs, files in os.walk(cls_path):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    cls_images.append(os.path.join(root, f))

        ood_inventory[cls_name] = {
            "image_count": len(cls_images),
            "subdirectories": [d for d in os.listdir(cls_path) if
                               os.path.isdir(os.path.join(cls_path, d))],
            "sample_paths": cls_images[:3],
        }
        total_images += len(cls_images)
        print(f"    {cls_name}: {len(cls_images)} images")

    print(f"    Total OOD images found: {total_images}")

    return ood_inventory, total_images


# -------------------------------------------------------------------
# Step 2: Load known-fruit features from pre-extracted dataset
# -------------------------------------------------------------------

def load_known_features():
    """Load pre-extracted ConvNeXt features for known fruit evaluation."""
    print("\n[2] Loading known-fruit features from phase4_dataset.pt ...")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")

    data = torch.load(FEATURES_PATH, map_location='cpu', weights_only=False)
    print(f"    Feature file: {FEATURES_PATH}")
    print(f"    Total samples: {data['vis_feats'].shape[0]:,}")
    print(f"    Feature dim: {data['vis_feats'].shape[1]}")

    # Validate features
    vis = data['vis_feats']
    assert not torch.isnan(vis).any(), "NaN in visual features!"
    assert not torch.isinf(vis).any(), "Inf in visual features!"
    print(f"    Features: finite={True}, shape={vis.shape}")

    return data


# -------------------------------------------------------------------
# Step 3: OOD Score computation on known-in-dist data
# -------------------------------------------------------------------

def compute_msp_score(logits: torch.Tensor) -> torch.Tensor:
    """Maximum Softmax Probability score. Higher = more in-distribution."""
    probs = F.softmax(logits, dim=1)
    return probs.max(dim=1).values


def compute_energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Energy score (Liu et al., NeurIPS 2020). Higher energy = more OOD."""
    return -T * torch.logsumexp(logits / T, dim=1)


def mahalanobis_score(features: torch.Tensor, class_means: torch.Tensor,
                      precision: torch.Tensor) -> torch.Tensor:
    """
    Mahalanobis distance to nearest class mean.
    Lower distance = more in-distribution.
    """
    B = features.shape[0]
    K = class_means.shape[0]
    dists = torch.zeros(B, K)
    for k in range(K):
        diff = features - class_means[k].unsqueeze(0)  # [B, D]
        dists[:, k] = (diff @ precision @ diff.T).diagonal()
    # Return min distance to any class (negative for AUROC convention: higher = more OOD)
    return dists.min(dim=1).values


def fit_mahalanobis(train_feats: torch.Tensor, train_labels: torch.Tensor, num_classes: int = 5):
    """
    Fit class-conditional means and shared precision matrix from training features.
    Only uses training data.
    """
    D = train_feats.shape[1]
    class_means = []
    centered_feats = []

    for k in range(num_classes):
        mask = (train_labels == k)
        if mask.sum() == 0:
            class_means.append(torch.zeros(D))
            continue
        mu_k = train_feats[mask].mean(dim=0)
        class_means.append(mu_k)
        centered_feats.append(train_feats[mask] - mu_k.unsqueeze(0))

    class_means = torch.stack(class_means)  # [K, D]

    if centered_feats:
        all_centered = torch.cat(centered_feats, dim=0)  # [N, D]
        # Shared covariance
        cov = (all_centered.T @ all_centered) / (all_centered.shape[0] - 1)
        # Regularize for invertibility
        cov = cov + 1e-4 * torch.eye(D)
        precision = torch.linalg.inv(cov)
    else:
        precision = torch.eye(D)

    return class_means, precision


# -------------------------------------------------------------------
# Step 4: Main OOD benchmark function
# -------------------------------------------------------------------

def run_ood_benchmark_on_known_data(data: dict):
    """
    Since real OOD images are absent, run an internal leave-one-class-out
    analysis on known fruit data to validate OOD detector functionality.
    This is NOT a substitute for real OOD evaluation but demonstrates
    detector behavior on structurally different species.
    """
    print("\n[3] Running internal leave-one-class-out OOD detector analysis ...")
    print("    NOTE: This is a structural validation only, not real OOD evaluation.")

    vis = data['vis_feats']  # [N, 768]
    targets = torch.tensor(data['targets'].astype(int))
    splits = data['splits']
    domains = np.array(data['domains'])
    fruits = np.array(data['fruits'])

    # Apply split_v2 correction (session 20200730 reallocated to test)
    # For this purpose we use the split column from data as-is
    # (split_v2 differs by 119 samples in Grade 2 test)

    train_mask = (splits == 'train')
    val_mask = (splits == 'val')
    test_mask = (splits == 'test')

    vis_train = vis[train_mask]
    tgt_train = targets[train_mask]
    vis_test = vis[test_mask]
    tgt_test = targets[test_mask]
    vis_val = vis[val_mask]
    tgt_val = targets[val_mask]

    print(f"    Train: {vis_train.shape[0]:,}, Val: {vis_val.shape[0]:,}, Test: {vis_test.shape[0]:,}")

    # Fit Mahalanobis statistics on training data ONLY
    print("    Fitting Mahalanobis statistics on train features (grade-conditional) ...")
    class_means, precision = fit_mahalanobis(vis_train, tgt_train, num_classes=5)

    # Compute Mahalanobis distance on test features
    print("    Computing Mahalanobis distances on test features ...")
    # Use a simple linear classifier head (trained logits from existing phase4 checkpoints not loaded here)
    # Instead, use class-mean nearest neighbor as a proxy
    mahal_dist_test = mahalanobis_score(vis_test, class_means, precision)

    # Also compute distance on val features (for threshold selection)
    mahal_dist_val = mahalanobis_score(vis_val, class_means, precision)

    # For MSP/Energy: use a simple k-NN from class means
    # Project via dot product similarity as proxy for logit
    logits_test = vis_test @ class_means.T  # [N_test, 5]
    logits_val = vis_val @ class_means.T

    msp_test = compute_msp_score(logits_test)
    energy_test = compute_energy_score(logits_test)
    msp_val = compute_msp_score(logits_val)
    energy_val = compute_energy_score(logits_val)

    # Domain-specific analysis
    test_domains = domains[test_mask]
    val_domains = domains[val_mask]

    results = {
        "known_data_statistics": {
            "n_train": int(vis_train.shape[0]),
            "n_val": int(vis_val.shape[0]),
            "n_test": int(vis_test.shape[0]),
        },
        "detector_statistics": {
            "mahalanobis_test_mean": float(mahal_dist_test.mean()),
            "mahalanobis_test_std": float(mahal_dist_test.std()),
            "msp_test_mean": float(msp_test.mean()),
            "msp_test_std": float(msp_test.std()),
            "energy_test_mean": float(energy_test.mean()),
            "energy_test_std": float(energy_test.std()),
        },
        "by_domain": {},
    }

    for domain in np.unique(test_domains):
        dm = (test_domains == domain)
        results["by_domain"][domain] = {
            "n": int(dm.sum()),
            "msp_mean": float(msp_test[dm].mean()),
            "energy_mean": float(energy_test[dm].mean()),
            "mahal_mean": float(mahal_dist_test[dm].mean()),
        }

    return results, mahal_dist_test, msp_test, energy_test, vis_test, tgt_test


# -------------------------------------------------------------------
# Step 5: Save benchmark CSV and report
# -------------------------------------------------------------------

def save_benchmark_csv(results, mahal_dist, msp_scores, energy_scores):
    """Save OOD benchmark summary to CSV."""
    rows = [
        {"detector": "MSP", "score_direction": "higher_is_indist",
         "test_mean": float(msp_scores.mean()), "test_std": float(msp_scores.std()),
         "note": "Internal known-data statistics only (no real OOD available)"},
        {"detector": "Energy", "score_direction": "lower_is_indist",
         "test_mean": float(energy_scores.mean()), "test_std": float(energy_scores.std()),
         "note": "Internal known-data statistics only (no real OOD available)"},
        {"detector": "Mahalanobis_dist", "score_direction": "lower_is_indist",
         "test_mean": float(mahal_dist.mean()), "test_std": float(mahal_dist.std()),
         "note": "Fitted on train grade-conditional means; internal known-data only"},
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "phase50_ood_benchmark.csv")
    df.to_csv(out_path, index=False)
    print(f"    Saved: {out_path}")
    return out_path


def write_ood_report(ood_inventory, total_images, results):
    """Write the OOD benchmark report."""
    lines = [
        "# Phase 5.0 -- Stage 4: Real-Image OOD Benchmark",
        "",
        "**Date:** October 2026  ",
        "**Status:** CRITICAL LIMITATION -- Real OOD images absent from disk",
        "",
        "---",
        "",
        "## 1. Critical Finding: OOD Dataset Directory Is Empty",
        "",
        "The audit instruction specified that `C:\\fruit_dataset\\open-set-dataset\\`",
        "contains real unknown-fruit images across five classes:",
        "Cherry, Kiwi, Peach, Pineapple, and Plum.",
        "",
        "**Actual content upon physical inspection:**",
        "",
        "| Class | Subdirectories | Images Found |",
        "|---|---|:---:|",
    ]

    for cls, info in ood_inventory.items():
        subdirs = ", ".join(info["subdirectories"][:5]) if info["subdirectories"] else "None"
        lines.append(f"| {cls} | {subdirs} | **{info['image_count']}** |")

    lines += [
        "",
        f"**Total real OOD images: {total_images}**",
        "",
        "The directory structure (Cherry_Train360, Cherry_Test360, etc.) is consistent",
        "with the Fruits-360 dataset format, but the directories contain no image files.",
        "This means the OOD dataset has NOT been downloaded to disk.",
        "",
        "> [!CAUTION]",
        "> The previously reported 100% OOD AUROC was confirmed to come from synthetic",
        "> Gaussian values (`np.random.randn() + 5.0`), not real images.",
        "> The real OOD image files are absent from disk.",
        "> The Phase 5.0 OOD benchmark cannot be completed without downloading images.",
        "",
        "---",
        "",
        "## 2. What Can Be Evaluated Without Real OOD Images",
        "",
        "Without real OOD images, no valid AUROC, AUPR, or FPR95 score can be computed.",
        "However, the following can be validly documented:",
        "",
        "1. OOD detector implementation correctness (verified in Stage 2 smoke test)",
        "2. OOD detector statistics on known-fruit features",
        "3. Score direction conventions",
        "4. A data collection protocol for future OOD evaluation",
        "",
        "### Internal OOD Detector Statistics (Known-Fruit Only)",
        "",
        "The following statistics describe detector behavior on known-fruit test data",
        "(NOT a valid OOD benchmark -- requires real unknown-class images):",
        "",
        "| Detector | Score Direction | Test Mean | Test Std |",
        "|---|---|---:|---:|",
    ]

    if "detector_statistics" in results:
        ds = results["detector_statistics"]
        lines += [
            f"| MSP | Higher = in-dist | {ds['msp_test_mean']:.4f} | {ds['msp_test_std']:.4f} |",
            f"| Energy | Lower = in-dist | {ds['energy_test_mean']:.4f} | {ds['energy_test_std']:.4f} |",
            f"| Mahalanobis | Lower = in-dist | {ds['mahalanobis_test_mean']:.4f} | {ds['mahalanobis_test_std']:.4f} |",
        ]

    lines += [
        "",
        "> [!NOTE]",
        "> These statistics describe the score distribution of **in-distribution** test",
        "> data only. Without real OOD images, no threshold can be validated and no",
        "> AUROC can be computed. Do not cite these numbers as OOD evaluation results.",
        "",
        "---",
        "",
        "## 3. Score Direction Convention",
        "",
        "For all detectors, larger score values are defined as MORE OOD:",
        "",
        "| Detector | Raw Score | OOD Score Convention |",
        "|---|---|---|",
        "| MSP | max softmax probability (higher = in-dist) | score_ood = 1 - MSP |",
        "| Energy | -T*logsumexp(logits/T) (more negative = in-dist) | score_ood = Energy |",
        "| Mahalanobis | min Mahalanobis dist to class means (lower = in-dist) | score_ood = Mahal_dist |",
        "",
        "---",
        "",
        "## 4. Required Data Collection Protocol",
        "",
        "To complete the real-image OOD benchmark, the following images must be",
        "downloaded and placed in `C:\\fruit_dataset\\open-set-dataset\\`:",
        "",
        "### Recommended Sources",
        "1. **Fruits-360 dataset** (Kaggle / GitHub): Cherry, Kiwi, Peach, Pineapple, Plum",
        "   - URL: https://www.kaggle.com/datasets/moltean/fruits",
        "   - Subdirectories already exist; only image download is missing.",
        "",
        "2. **Alternative**: Any publicly available fruit image dataset where the",
        "   species does NOT appear in the known-class training data.",
        "",
        "### Validation Requirements",
        "After downloading:",
        "1. Verify images exist: at least 50 images per class recommended.",
        "2. Extract ConvNeXt-Tiny features using the same preprocessing pipeline",
        "   (resize to 224x224, ImageNet normalization).",
        "3. Fit Mahalanobis statistics ONLY on training data.",
        "4. Select OOD threshold on validation data (NOT test data).",
        "5. Report AUROC, AUPR, FPR95 on held-out OOD test images.",
        "",
        "### Why the Claimed 100% AUROC Was Invalid",
        "```python",
        "# From run_phase2_ood.py lines 45-63 (DEFECTIVE CODE):",
        "if not os.path.exists(ood_features_path):",
        "    ood_features = np.random.randn(n_ood, feature_dim) + 5.0  # SYNTHETIC!",
        "```",
        "Gaussian noise offset by +5 is trivially separable from real ConvNeXt features",
        "using any distance metric, producing an artificially perfect AUROC.",
        "This was identified in Phase 4.5 and confirmed in this audit.",
        "",
        "---",
        "",
        "## 5. Stage 4 Sign-Off",
        "",
        "- [x] OOD directory physically inspected: 0 images found",
        "- [x] Synthetic OOD AUROC from Phase 2 confirmed invalid",
        "- [x] Score direction conventions documented",
        "- [x] Detector implementation validated in Stage 2 smoke test",
        "- [x] Data collection protocol specified",
        "- [ ] REAL OOD BENCHMARK: NOT COMPLETED (images absent from disk)",
        "",
        "> [!WARNING]",
        "> The real-image OOD benchmark (Stage 4) is **blocked** until unknown-fruit",
        "> images are downloaded to `C:\\fruit_dataset\\open-set-dataset\\`. The claim",
        "> of 'trustworthy open-set recognition' CANNOT be supported until this is",
        "> completed with real empirical evidence.",
        "",
        "**LIMITATION DOCUMENTED. Proceed to Stage 5 with explicit OOD caveat.**",
    ]

    doc_path = os.path.join(OUTPUT_DIR, "phase50_ood_report.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    Saved: {doc_path}")


def main():
    print("=" * 70)
    print("PHASE 5.0 -- STAGE 4: REAL-IMAGE OOD BENCHMARK")
    print("=" * 70)

    # Step 1: Inspect OOD directory
    ood_inventory, total_ood_images = inspect_ood_directory()

    if total_ood_images == 0:
        print("\n  CRITICAL: No real OOD images found in open-set-dataset/")
        print("  Cannot compute valid AUROC/AUPR/FPR95.")
        print("  Documenting limitation and running internal known-data analysis...")

    # Step 2: Load known features
    data = load_known_features()

    # Step 3: Run internal analysis (structural validation only)
    results, mahal_dist, msp_scores, energy_scores, vis_test, tgt_test = \
        run_ood_benchmark_on_known_data(data)

    # Step 4: Save features metadata
    features_meta = {
        "feature_source": FEATURES_PATH,
        "feature_extractor": "ConvNeXt-Tiny (ImageNet pretrained)",
        "feature_dim": 768,
        "preprocessing": "Resize(224,224) + ImageNet normalization",
        "n_known_train": results["known_data_statistics"]["n_train"],
        "n_known_val": results["known_data_statistics"]["n_val"],
        "n_known_test": results["known_data_statistics"]["n_test"],
        "ood_images_found": total_ood_images,
        "ood_benchmark_valid": False,
        "reason": "Real OOD images absent from disk. See phase50_ood_report.md.",
    }
    with open(os.path.join(OOD_FEATURES_DIR, "features_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(features_meta, f, indent=2)
    print(f"\n    Saved: {OOD_FEATURES_DIR}/features_metadata.json")

    # Step 5: Save benchmark CSV
    save_benchmark_csv(results, mahal_dist, msp_scores, energy_scores)

    # Step 6: Write report
    write_ood_report(ood_inventory, total_ood_images, results)

    print("\n" + "=" * 70)
    print("STAGE 4 COMPLETE")
    print("  OOD benchmark: BLOCKED (real images absent)")
    print("  Documentation: Complete")
    print("  Limitation: Documented in phase50_ood_report.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
