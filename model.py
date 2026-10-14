"""
model.py
========
ConvNeXt-Tiny Architecture for 5-Class Species Classification AND Phase 3 CORAL Ordinal Regression.

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Features:
- ConvNeXtFruitClassifier: 5-Class Species Classification Head
- CoralHead: Rank-Consistent Ordinal Regression Head (Cao et al., 2020)
- ConvNeXtCoralClassifier: Joint / Dedicated Ordinal Model for 1st, 2nd, 3rd Grade
"""

import torch
import torch.nn as nn
import timm


class ConvNeXtFruitClassifier(nn.Module):
    """ConvNeXt-Tiny species classifier wrapper."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        self.num_features = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)


class CoralHead(nn.Module):
    """
    Rank-Consistent Ordinal Regression (CORAL) Head (Cao, Mirjalili, & Raschka, 2020).
    
    For K ordinal ranks, computes K-1 binary classification tasks sharing a single weight
    vector w and rank-specific bias thresholds b_0, b_1, ..., b_{K-2}.
    """

    def __init__(self, in_features: int = 768, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1  # K-1 tasks

        # Shared weight vector w: (in_features -> 1)
        self.weight = nn.Linear(in_features, 1, bias=False)
        # Task-specific thresholds b_k
        self.bias = nn.Parameter(torch.zeros(self.num_tasks))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (N, in_features)
        Returns rank logits of shape (N, K-1): g(x) + b_k
        """
        linear_out = self.weight(features)  # (N, 1)
        logits = linear_out + self.bias     # Broadcasts to (N, K-1)
        return logits


class ConvNeXtCoralClassifier(nn.Module):
    """
    ConvNeXt-Tiny backbone equipped with Rank-Consistent Ordinal Regression (CORAL) head
    for fruit grade estimation (1st Grade < 2nd Grade < 3rd Grade).
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True, drop_rate: float = 0.15):
        super().__init__()
        self.num_classes = num_classes

        # ConvNeXt feature extractor
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,  # Pooling layer output (768-dim)
            drop_rate=drop_rate,
        )
        self.num_features = self.backbone.num_features
        self.norm = nn.LayerNorm(self.num_features)
        self.drop = nn.Dropout(drop_rate)

        # CORAL Ordinal Head
        self.head = CoralHead(in_features=self.num_features, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns CORAL rank logits of shape (N, K-1).
        Logits > 0 correspond to probabilities P(y > k) > 0.5.
        """
        feats = self.backbone(x)
        feats = self.norm(feats)
        feats = self.drop(feats)
        return self.head(feats)

    def predict_rank(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decodes CORAL logits of shape (N, K-1) into discrete ordinal ranks in {0, 1, ..., K-1}.
        hat_y = sum_k (sigmoid(logits_k) > 0.5)
        """
        probs = torch.sigmoid(logits)
        predicted_ranks = (probs > 0.5).sum(dim=1)
        return predicted_ranks


def build_convnext_model(num_classes: int = 5, pretrained: bool = True, device: str = "cuda") -> nn.Module:
    model = ConvNeXtFruitClassifier(num_classes=num_classes, pretrained=pretrained)
    return model.to(device)


def build_coral_model(num_classes: int = 3, pretrained: bool = True, device: str = "cuda") -> nn.Module:
    model = ConvNeXtCoralClassifier(num_classes=num_classes, pretrained=pretrained)
    return model.to(device)
