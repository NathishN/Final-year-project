"""
loss.py
=======
Loss Functions for Class-Imbalanced Classification AND Phase 3 CORAL Ordinal Regression.

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Included Losses:
1. ClassBalancedFocalLoss (beta=0.9999, gamma=2.0)
2. FocalLoss (gamma=2.0)
3. LabelSmoothedCrossEntropy
4. CoralLoss (Cao et al., Pattern Recognition Letters 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def label_to_coral_targets(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """
    Converts discrete ordinal rank targets y in {0, 1, ..., K-1} into K-1 binary indicator targets.
    Example for K=3:
      y = 0 (1st Grade) -> [0, 0]
      y = 1 (2nd Grade) -> [1, 0]
      y = 2 (3rd Grade) -> [1, 1]
    """
    num_tasks = num_classes - 1
    batch_size = labels.size(0)
    coral_targets = torch.zeros((batch_size, num_tasks), dtype=torch.float32, device=labels.device)

    for k in range(num_tasks):
        coral_targets[:, k] = (labels > k).float()

    return coral_targets


class CoralLoss(nn.Module):
    """
    Rank-Consistent Ordinal Regression Loss (CORAL, Cao et al., 2020).
    Computes sum of Binary Cross-Entropy losses across K-1 binary sub-tasks.
    """

    def __init__(self, num_classes: int = 3, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : CORAL logits of shape (N, K-1)
        targets: Ordinal rank targets of shape (N,) in {0, 1, ..., K-1}
        """
        coral_targets = label_to_coral_targets(targets, self.num_classes)
        
        # Binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, coral_targets, reduction="none")
        task_sum = torch.sum(bce_loss, dim=1)  # Sum across K-1 tasks

        if self.reduction == "mean":
            return task_sum.mean()
        elif self.reduction == "sum":
            return task_sum.sum()
        else:
            return task_sum


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_preds).scatter_(
                1, targets.unsqueeze(1), 1.0 - self.label_smoothing
            )
            smooth_targets += self.label_smoothing / num_classes

        loss = -torch.sum(smooth_targets * log_preds, dim=-1)

        if self.weight is not None:
            if self.weight.device != inputs.device:
                self.weight = self.weight.to(inputs.device)
            sample_weights = self.weight[targets]
            loss = loss * sample_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, label_smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(log_probs).scatter_(
                1, targets.unsqueeze(1), 1.0 - self.label_smoothing
            )
            smooth_targets += self.label_smoothing / num_classes

        pt = torch.sum(smooth_targets * probs, dim=-1)
        focal_weight = (1.0 - pt) ** self.gamma
        ce_loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha[targets]
            focal_loss = focal_loss * alpha_weight

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_counts: list, beta: float = 0.9999, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_counts = class_counts

        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(class_counts)
        self.alpha = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        probs = F.softmax(inputs, dim=-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(log_probs).scatter_(
                1, targets.unsqueeze(1), 1.0 - self.label_smoothing
            )
            smooth_targets += self.label_smoothing / num_classes

        pt = torch.sum(smooth_targets * probs, dim=-1)
        focal_weight = (1.0 - pt) ** self.gamma
        ce_loss = -torch.sum(smooth_targets * log_probs, dim=-1)

        alpha_weight = self.alpha[targets]
        loss = focal_weight * ce_loss * alpha_weight
        return loss.mean()


def get_loss_function(
    loss_name: str,
    class_counts: list = None,
    label_smoothing: float = 0.1,
    device: str = "cuda"
) -> nn.Module:
    loss_name = loss_name.lower()

    if loss_name == "coral":
        return CoralLoss(num_classes=3)
    elif loss_name == "cb_focal":
        assert class_counts is not None, "class_counts required for CB-Focal Loss"
        return ClassBalancedFocalLoss(class_counts=class_counts, beta=0.9999, gamma=2.0, label_smoothing=label_smoothing)
    elif loss_name == "focal":
        return FocalLoss(gamma=2.0, label_smoothing=label_smoothing)
    elif loss_name == "ce":
        return LabelSmoothedCrossEntropy(label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")
