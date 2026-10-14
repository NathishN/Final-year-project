"""
severity_loss.py
================
Loss Functions and Evaluation Metrics for Phase 4 Context-Aware Severity Modeling (CASM).

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Includes:
1. CORALLoss (Rank-Consistent Ordinal Regression Loss)
2. CORNLoss (Conditional Ordinal Regression Loss)
3. FocalLoss (Multi-class / Ordinal Focal Loss)
4. StandardCELoss (Weighted Cross-Entropy Loss)
5. Metric computation utility functions (MAE, QWK, Off-by-one Accuracy, Macro-F1, Weighted-F1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_recall_fscore_support


# ─── 1. CORAL Loss (Rank-Consistent Ordinal Regression) ─────────────────────────

class CORALLoss(nn.Module):
    """
    CORAL Loss (Cao et al., 2020).
    Converts rank label y in {0, 1, ..., K-1} into K-1 binary targets:
        y_binary[k] = 1 if y > k else 0
    Computes binary cross entropy across all K-1 binary task heads.
    """
    def __init__(self, num_classes: int = 5, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, K-1) raw logits from CORAL output layer.
            targets: (B,) ordinal rank labels in {0, ..., K-1}.
        Returns:
            Scalar loss tensor.
        """
        # Construct binary targets (B, K-1)
        # binary_targets[i, k] = 1.0 if targets[i] > k else 0.0
        levels = torch.arange(self.num_tasks, device=targets.device).unsqueeze(0)  # (1, K-1)
        binary_targets = (targets.unsqueeze(1) > levels).float()                     # (B, K-1)

        bce_loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='none')

        if self.class_weights is not None:
            # Weight loss by instance class weight
            sample_weights = self.class_weights[targets].unsqueeze(1)
            bce_loss = bce_loss * sample_weights

        return bce_loss.mean()


# ─── 2. CORN Loss (Conditional Ordinal Regression) ─────────────────────────────

class CORNLoss(nn.Module):
    """
    CORN Loss (Shi et al., 2023).
    Conditional Ordinal Regression Neural Network loss.
    Computes conditional binary cross entropy for ordered rank predictions.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, K-1) conditional binary logits.
            targets: (B,) ordinal rank targets.
        """
        levels = torch.arange(self.num_tasks, device=targets.device).unsqueeze(0)  # (1, K-1)
        binary_targets = (targets.unsqueeze(1) > levels).float()

        # Conditional mask: task k is only active for samples with target >= k
        mask = (targets.unsqueeze(1) >= levels).float()

        bce = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='none')
        masked_bce = bce * mask
        loss = masked_bce.sum() / (mask.sum() + 1e-8)
        return loss


# ─── 3. Focal Loss ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss (Lin et al., 2017) to address severe class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, num_classes: int = 5, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        pt = (probs * targets_onehot).sum(dim=1) + 1e-8
        log_pt = torch.log(pt)
        
        focal_weight = (1.0 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_loss = -alpha_weight * focal_weight * log_pt
        else:
            focal_loss = -focal_weight * log_pt

        return focal_loss.mean()


# ─── 4. Standard Cross-Entropy Loss ─────────────────────────────────────────────

class StandardCELoss(nn.Module):
    def __init__(self, num_classes: int = 5, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


# ─── 5. Ordinal Evaluation Metrics Helper ───────────────────────────────────────

def compute_severity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes full set of evaluation metrics required for IEEE paper evaluation:
    - MAE (Mean Absolute Error)
    - QWK (Quadratic Weighted Kappa)
    - Accuracy (%)
    - Off-by-one Tolerance Accuracy (%)
    - Macro-F1 (%)
    - Weighted-F1 (%)
    - Macro-Precision (%)
    - Macro-Recall (%)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    off_by_one = float(np.mean(np.abs(y_true - y_pred) <= 1)) * 100.0

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "MAE": round(mae, 4),
        "QWK": round(qwk, 4),
        "Accuracy": round(acc, 2),
        "Off_By_One_Acc": round(off_by_one, 2),
        "Macro_F1": round(f1_macro * 100.0, 2),
        "Weighted_F1": round(f1_weighted * 100.0, 2),
        "Macro_Precision": round(prec_macro * 100.0, 2),
        "Macro_Recall": round(rec_macro * 100.0, 2),
        "Total_Samples": len(y_true),
    }
