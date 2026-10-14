"""
casm_v2_model.py
================
CASM v2: Selective Species & OOD Fusion Network for Ordinal Fruit Severity Grading.

Architecture Specification:
1. Visual Branch: 768-D ConvNeXt-Tiny feature vector projected via 2-layer MLP to 256-D.
2. Species Branch: 16-D categorical embedding for fruit species (handles known indices 0..4 and optional unknown/predicted species).
3. OOD Branch: 2-D Mahalanobis distance & calibrated score representation (fitted strictly on training set).
4. Selective Fusion: Direct concatenation of active representation streams [256 + 16 + 2 = 274D] (removes FiLM & cross-attention noise).
5. Head: Rank-consistent 5-class CORAL Ordinal Head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CORALHead(nn.Module):
    """
    Rank-Consistent CORAL Output Head (Cao et al., 2020).
    Shared linear projection vector W in R^{1 x D} and K-1 unconstrained threshold biases b_k.
    """
    def __init__(self, in_features: int = 128, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1

        self.weight = nn.Parameter(torch.Tensor(1, in_features))
        self.bias   = nn.Parameter(torch.zeros(self.num_tasks))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> logits: (B, K-1)
        return torch.matmul(x, self.weight.t()) + self.bias

    def predict_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes monotonic cumulative rank probabilities."""
        return torch.sigmoid(logits)

    def predict_rank(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts logits to ordinal ranks in {0, ..., K-1} via thresholding > 0.5."""
        probs = self.predict_probs(logits)
        return (probs > 0.5).sum(dim=1)


class CORALLoss(nn.Module):
    """
    Task-binary cross-entropy loss for rank-consistent ordinal regression with class weighting.
    """
    def __init__(self, num_classes: int = 5, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_tasks = num_classes - 1
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: (B,) long tensor in {0, ..., K-1}
        levels = torch.arange(self.num_tasks, device=targets.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > levels).float()
        loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction="none")
        if self.class_weights is not None:
            weights = self.class_weights[targets].unsqueeze(1)
            loss = loss * weights
        return loss.mean()


class CASMv2Network(nn.Module):
    """
    CASM v2 Architecture with configurable context streams.
    """
    def __init__(
        self,
        vis_in_dim: int = 768,
        num_species: int = 5,
        num_classes: int = 5,
        use_species: bool = True,
        use_ood: bool = True,
        use_grade: bool = False,
        use_unc: bool = False
    ):
        super().__init__()
        self.vis_in_dim = vis_in_dim
        self.num_species = num_species
        self.num_classes = num_classes
        self.use_species = use_species
        self.use_ood = use_ood
        self.use_grade = use_grade
        self.use_unc = use_unc

        # Visual Projection Stream
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        fused_dim = 256
        if self.use_species:
            # 5 known species (indices 0..4) + 1 fallback for unknown species (index 5)
            self.species_embed = nn.Embedding(num_species + 1, 16)
            fused_dim += 16

        if self.use_ood:
            fused_dim += 2

        if self.use_grade:
            fused_dim += 4

        if self.use_unc:
            fused_dim += 2

        # Fusion MLP Bottleneck
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Ordinal Head
        self.head = CORALHead(in_features=128, num_classes=num_classes)

    def forward(
        self,
        vis: torch.Tensor,
        species: torch.Tensor = None,
        ood: torch.Tensor = None,
        grade: torch.Tensor = None,
        unc: torch.Tensor = None
    ) -> torch.Tensor:
        parts = [self.vis_proj(vis)]

        if self.use_species:
            if species is None:
                # Default to index 5 (unknown) if missing
                species = torch.full((vis.shape[0],), 5, dtype=torch.long, device=vis.device)
            else:
                # Clamp out-of-range species to index 5
                species = torch.clamp(species, 0, self.num_species)
            parts.append(self.species_embed(species))

        if self.use_ood:
            assert ood is not None, "OOD tensor required when use_ood=True"
            parts.append(ood)

        if self.use_grade:
            assert grade is not None, "Grade tensor required when use_grade=True"
            parts.append(grade)

        if self.use_unc:
            assert unc is not None, "Uncertainty tensor required when use_unc=True"
            parts.append(unc)

        fused = torch.cat(parts, dim=1)
        bottleneck = self.fusion_mlp(fused)
        logits = self.head(bottleneck)
        return logits

    def predict_rank(
        self,
        vis: torch.Tensor,
        species: torch.Tensor = None,
        ood: torch.Tensor = None,
        grade: torch.Tensor = None,
        unc: torch.Tensor = None
    ) -> torch.Tensor:
        logits = self.forward(vis, species, ood, grade, unc)
        return self.head.predict_rank(logits)
