"""
fruit_quality_dataset.py
========================
PyTorch Dataset class for:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading"

Supports:
- Filtering by: split, domain, fruit, label_type, is_ood
- Three separate label heads: grade, quality, severity
- Per-domain normalization statistics
- OOD flag for open-set evaluation
- Automatic class-weight computation for focal / weighted CE loss

Usage:
    from fruit_quality_dataset import FruitQualityDataset, get_transforms

    train_ds = FruitQualityDataset(
        csv_path   = "master_dataset.csv",
        split      = "train",
        label_type = "grade",
        domain     = "real_grade",
    )
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    for imgs, labels, meta in loader:
        ...
"""

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ─── Per-domain normalisation statistics ──────────────────────────────────────
# fruits360: ImageNet stats are appropriate (similar studio conditions)
# real_*:    Approximate stats from real-world Android camera images
DOMAIN_NORM_STATS = {
    "fruits360":    {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "real_grade":   {"mean": (0.500, 0.455, 0.405), "std": (0.248, 0.232, 0.222)},
    "real_quality": {"mean": (0.500, 0.455, 0.405), "std": (0.248, 0.232, 0.222)},
    "real_severity":{"mean": (0.480, 0.440, 0.395), "std": (0.242, 0.222, 0.212)},
}

# ─── Label vocabularies ────────────────────────────────────────────────────────
GRADE_MAP = {
    "grade_1": 0,   # best / 1st grade
    "grade_2": 1,
    "grade_3": 2,   # worst / 3rd grade
}

QUALITY_MAP = {
    "good":  0,
    "mixed": 1,
    "bad":   2,
}

SEVERITY_MAP = {
    "fresh":  0,
    "mild":   1,
    "rotten": 2,
}

FRUIT_MAP = {
    "Apple":  0,
    "Banana": 1,
    "Mango":  2,
    "Orange": 3,
    "Tomato": 4,
}

LABEL_MAPS = {
    "grade":    GRADE_MAP,
    "quality":  QUALITY_MAP,
    "severity": SEVERITY_MAP,
    "variety":  None,   # dynamic — built from dataframe
    "fruit":    FRUIT_MAP,
}

LABEL_COL = {
    "grade":    "grade_label_normalized",
    "quality":  "quality_label_normalized",
    "severity": "severity_label_normalized",
    "variety":  "variety",
    "fruit":    "fruit",
}


# ─── Transform factories ───────────────────────────────────────────────────────

def get_transforms(split: str, domain: str = "real_grade", img_size: int = 224):
    """
    Returns appropriate torchvision transform for a given split + domain.
    Fruits-360 images are 100x100 and already clean — lighter augmentation.
    Real-world images benefit from stronger augmentation.
    """
    stats = DOMAIN_NORM_STATS.get(domain, DOMAIN_NORM_STATS["real_grade"])
    norm = T.Normalize(mean=stats["mean"], std=stats["std"])

    if split == "train":
        if domain == "fruits360":
            # Lighter aug: already rotation-invariant from dataset construction
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                norm,
            ])
        else:
            # Stronger aug for real-world images
            return T.Compose([
                T.Resize((int(img_size * 1.15), int(img_size * 1.15))),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                norm,
            ])
    else:
        # val / test: deterministic
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            norm,
        ])


# ─── Dataset class ─────────────────────────────────────────────────────────────

class FruitQualityDataset(Dataset):
    """
    Multi-label fruit quality dataset loaded from master_dataset.csv.

    Parameters
    ----------
    csv_path    : str   — path to master_dataset.csv
    split       : str   — "train" | "val" | "test" | "OOD_TEST"
    label_type  : str   — "grade" | "quality" | "severity" | "variety"
    domain      : str   — filter to specific domain (None = all domains)
    fruits      : list  — filter to specific fruits (None = all)
    include_ood : bool  — if True, also include OOD rows (split=OOD_TEST)
    ood_only    : bool  — if True, return ONLY OOD rows
    transform         — torchvision transform (None = auto from split+domain)
    img_size    : int   — resize target
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        label_type: str = "grade",
        domain: str = None,
        fruits: list = None,
        include_ood: bool = False,
        ood_only: bool = False,
        transform=None,
        img_size: int = 224,
    ):
        assert label_type in LABEL_MAPS, f"label_type must be one of {list(LABEL_MAPS.keys())}"
        self.label_type = label_type
        self.img_size   = img_size
        self.split      = split

        # ── Load CSV ──────────────────────────────────────────────────────────
        df = pd.read_csv(csv_path, dtype=str)

        # Drop EXCLUDED rows
        df = df[df["split"] != "EXCLUDED"]

        # ── Filter by split / OOD ─────────────────────────────────────────────
        if ood_only:
            df = df[df["is_ood"].str.lower() == "true"]
        elif include_ood:
            df = df[(df["split"] == split) | (df["is_ood"].str.lower() == "true")]
        else:
            df = df[df["split"] == split]
            df = df[df["is_ood"].str.lower() == "false"]

        # ── Filter by domain ──────────────────────────────────────────────────
        if domain is not None:
            df = df[df["domain"] == domain]

        # ── Filter by fruits ──────────────────────────────────────────────────
        if fruits is not None:
            df = df[df["fruit"].isin(fruits)]

        # ── Filter rows that have a valid label for the requested label_type ──
        col = LABEL_COL[label_type]
        df = df[df[col].notna() & (df[col].str.strip() != "")]

        if len(df) == 0:
            raise ValueError(
                f"No samples found for split='{split}', label_type='{label_type}', "
                f"domain='{domain}', fruits={fruits}. "
                "Check your master_dataset.csv filters."
            )

        self.df = df.reset_index(drop=True)

        # ── Build label map for 'variety' (dynamic) ───────────────────────────
        if label_type == "variety":
            unique_varieties = sorted(self.df["variety"].unique())
            self.label_map = {v: i for i, v in enumerate(unique_varieties)}
        else:
            self.label_map = LABEL_MAPS[label_type]

        self.num_classes = len(self.label_map)

        # ── Transform ─────────────────────────────────────────────────────────
        if transform is None:
            # Use the most common domain in this subset for transform selection
            dominant_domain = self.df["domain"].value_counts().idxmax()
            self.transform = get_transforms(split, dominant_domain, img_size)
        else:
            self.transform = transform

        print(
            f"[FruitQualityDataset] split={split}, label_type={label_type}, "
            f"domain={domain}, fruits={fruits} -> {len(self.df):,} samples, "
            f"{self.num_classes} classes"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load image ────────────────────────────────────────────────────────
        img_path = row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Cannot open image: {img_path}\nError: {e}")

        if self.transform:
            img = self.transform(img)

        # ── Label ─────────────────────────────────────────────────────────────
        col        = LABEL_COL[self.label_type]
        label_str  = row[col]
        label      = self.label_map.get(label_str, -1)
        if label == -1:
            raise KeyError(f"Unknown label '{label_str}' for label_type='{self.label_type}'")
        label = torch.tensor(label, dtype=torch.long)

        # ── Fruit class index ─────────────────────────────────────────────────
        fruit_idx = FRUIT_MAP.get(row["fruit"], -1)

        # ── Metadata dict ─────────────────────────────────────────────────────
        meta = {
            "image_path":   img_path,
            "fruit":        row["fruit"],
            "fruit_idx":    torch.tensor(fruit_idx, dtype=torch.long),
            "domain":       row["domain"],
            "is_ood":       row["is_ood"].lower() == "true",
            "session_id":   row.get("session_id", ""),
            "label_str":    label_str,
        }

        return img, label, meta

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted CE or focal loss.
        Returns tensor of shape (num_classes,).
        """
        col = LABEL_COL[self.label_type]
        counts = self.df[col].map(self.label_map).value_counts().sort_index()
        # Fill in any missing classes with 1 (avoid division by zero)
        freq = torch.zeros(self.num_classes)
        for cls_idx, cnt in counts.items():
            freq[cls_idx] = cnt
        freq = torch.where(freq == 0, torch.ones_like(freq), freq)
        weights = 1.0 / freq
        weights = weights / weights.sum() * self.num_classes  # normalise
        return weights

    def class_distribution(self):
        """Print class distribution for this split."""
        col = LABEL_COL[self.label_type]
        dist = self.df[col].value_counts().sort_index()
        print(f"\nClass distribution [{self.label_type}, split={self.split}]:")
        for label, count in dist.items():
            idx = self.label_map.get(label, "?")
            pct = count / len(self.df) * 100
            bar = "█" * int(pct / 2)
            print(f"  [{idx}] {label:<12}: {count:>5,} ({pct:5.1f}%) {bar}")
    def get_domain_balanced_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        """
        Creates a WeightedRandomSampler based on (fruit x domain) stratum counts.
        Prevents model from over-indexing on Fruits-360 studio rotations and ensures
        minority real-world domains (real_quality, real_severity, real_grade) are sampled
        with higher probability.
        """
        stratums = self.df["fruit"] + "_" + self.df["domain"]
        counts = stratums.value_counts()
        weights = 1.0 / self.df.index.map(lambda i: counts[stratums.iloc[i]])
        sample_weights = torch.tensor(weights.values, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler


# ─── DataLoader factories ─────────────────────────────────────────────────────

def build_grade_loaders(csv_path: str, fruits=None, batch_size=32, num_workers=4, img_size=224):
    """Build train/val/test loaders for the ordinal grade experiment."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = FruitQualityDataset(
            csv_path   = csv_path,
            split      = split,
            label_type = "grade",
            domain     = "real_grade",
            fruits     = fruits or ["Apple", "Mango", "Tomato"],  # Banana excluded (3rd grade broken)
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
        )
    return loaders


def build_quality_loaders(csv_path: str, fruits=None, batch_size=32, num_workers=4, img_size=224):
    """Build train/val/test loaders for the quality classification experiment."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = FruitQualityDataset(
            csv_path   = csv_path,
            split      = split,
            label_type = "quality",
            domain     = "real_quality",
            fruits     = fruits or ["Apple", "Banana", "Orange"],
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
        )
    return loaders


def build_severity_loaders(csv_path: str, batch_size=32, num_workers=4, img_size=224):
    """Build train/val/test loaders for the severity modeling experiment."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = FruitQualityDataset(
            csv_path   = csv_path,
            split      = split,
            label_type = "severity",
            domain     = "real_severity",
            fruits     = ["Banana", "Tomato"],   # only fruits with severity labels
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
        )
    return loaders


def build_fruits360_loaders(csv_path: str, batch_size=64, num_workers=4, img_size=224):
    """Build train/test loaders for Fruits-360 species classification + open-set."""
    loaders = {}
    for split in ("train", "test"):
        ds = FruitQualityDataset(
            csv_path   = csv_path,
            split      = split,
            label_type = "variety",
            domain     = "fruits360",
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
        )
    return loaders


# ─── Quick test / sanity check ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    csv_path = os.path.join(r"C:\fruit_dataset", "master_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] master_dataset.csv not found at {csv_path}")
        print("Run generate_master_csv.py first.")
        sys.exit(1)

    print("=== Sanity check: Grade dataset ===")
    ds = FruitQualityDataset(csv_path, split="train", label_type="grade", domain="real_grade")
    ds.class_distribution()
    print(f"Class weights: {ds.get_class_weights()}")
    img, label, meta = ds[0]
    print(f"Sample: img.shape={img.shape}, label={label.item()}, fruit={meta['fruit']}")

    print("\n=== Sanity check: Quality dataset ===")
    ds2 = FruitQualityDataset(csv_path, split="train", label_type="quality", domain="real_quality")
    ds2.class_distribution()
    print(f"Class weights (for focal loss): {ds2.get_class_weights()}")

    print("\n=== Sanity check: Severity dataset ===")
    ds3 = FruitQualityDataset(csv_path, split="train", label_type="severity", domain="real_severity")
    ds3.class_distribution()

    print("\nAll checks passed.")
