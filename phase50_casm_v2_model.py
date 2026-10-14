"""
phase50_casm_v2_model.py
========================
Phase 5.0 Corrected Model Architectures with Monotonic CORAL Head
and Robust Species Embedding.

Contains:
1. MonotonicCORALHead: Guarantees monotonic cutpoints and valid probabilities.
2. RobustSpeciesEmbedding: Strict boundary checking and reserved UNKNOWN index (5).
3. CASMv2Model: Flexible model supporting Configurations A, B, C, D:
     - Config A: Image only
     - Config B: Image + Species
     - Config C: Image + OOD
     - Config D: Image + Species + OOD
4. Loss computation with support for unweighted, inverse, and Effective Number weights.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicCORALHead(nn.Module):
    """
    Rank-Consistent CORAL Head with strictly decreasing cutpoint biases.
    Logits: logits_k = x @ W^T + b_k, where b_k = theta_0 - sum_{j=1}^k softplus(delta_j).
    Guarantees:
      - logits_0 > logits_1 > ... > logits_{K-2}
      - Cumulative probs P(Y > k) are strictly decreasing
      - Class probs P(Y = k) are non-negative and sum to 1.0
    """
    def __init__(self, in_features: int = 128, num_classes: int = 5):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1

        self.weight = nn.Parameter(torch.Tensor(1, in_features))
        self.theta_0 = nn.Parameter(torch.zeros(1))
        if self.num_tasks > 1:
            self.raw_deltas = nn.Parameter(torch.ones(self.num_tasks - 1) * 0.5413)  # softplus ~= 1.0
        else:
            self.raw_deltas = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def get_biases(self) -> torch.Tensor:
        if self.num_tasks == 1:
            return self.theta_0
        deltas = F.softplus(self.raw_deltas)
        cum_deltas = torch.cumsum(deltas, dim=0)
        tail_biases = self.theta_0 - cum_deltas
        return torch.cat([self.theta_0, tail_biases])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_base = torch.matmul(x, self.weight.t())  # [B, 1]
        biases = self.get_biases()                      # [K-1]
        return logits_base + biases

    def predict_cumulative_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def predict_class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        cum_probs = self.predict_cumulative_probs(logits)  # [B, K-1]
        B = logits.shape[0]
        p0 = (1.0 - cum_probs[:, 0:1]).clamp(min=0.0, max=1.0)
        if self.num_tasks > 1:
            pk = (cum_probs[:, :-1] - cum_probs[:, 1:]).clamp(min=0.0, max=1.0)
        else:
            pk = torch.empty((B, 0), device=logits.device)
        pK = cum_probs[:, -1:].clamp(min=0.0, max=1.0)
        probs = torch.cat([p0, pk, pK], dim=1)
        sums = probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return probs / sums

    def predict_rank(self, logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)


class RobustSpeciesEmbedding(nn.Module):
    """
    Robust Species Embedding with reserved UNKNOWN_SPECIES index (5).
    Safely clamps negative, missing, or out-of-range IDs to UNKNOWN.
    """
    def __init__(self, num_known_species: int = 5, embedding_dim: int = 16):
        super().__init__()
        self.num_known_species = num_known_species
        self.unknown_idx = num_known_species
        self.embedding = nn.Embedding(num_known_species + 1, embedding_dim)

    def forward(self, species_ids: torch.Tensor) -> torch.Tensor:
        clean_ids = species_ids.clone().long()
        invalid_mask = (clean_ids < 0) | (clean_ids >= self.num_known_species)
        clean_ids[invalid_mask] = self.unknown_idx
        return self.embedding(clean_ids)


class CASMv2Model(nn.Module):
    """
    Phase 5.0 CASM v2 architecture supporting configurations A, B, C, D.
    """
    def __init__(
        self,
        vis_dim: int = 768,
        use_species: bool = False,
        use_ood: bool = False,
        species_dim: int = 16,
        ood_dim: int = 2,
        hidden_dim: int = 256,
        fc_dim: int = 128,
        num_classes: int = 5
    ):
        super().__init__()
        self.use_species = use_species
        self.use_ood = use_ood

        # Visual projection
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        fused_dim = hidden_dim
        if use_species:
            self.species_emb = RobustSpeciesEmbedding(num_known_species=5, embedding_dim=species_dim)
            fused_dim += species_dim
        if use_ood:
            fused_dim += ood_dim

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, fc_dim),
            nn.LayerNorm(fc_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Monotonic CORAL Head
        self.head = MonotonicCORALHead(in_features=fc_dim, num_classes=num_classes)

    def forward(self, vis: torch.Tensor, species: torch.Tensor = None, ood: torch.Tensor = None) -> torch.Tensor:
        f_vis = self.vis_proj(vis)
        parts = [f_vis]
        if self.use_species:
            if species is None:
                species = torch.full((vis.shape[0],), 5, dtype=torch.long, device=vis.device)
            parts.append(self.species_emb(species))
        if self.use_ood:
            if ood is None:
                ood = torch.zeros((vis.shape[0], 2), dtype=torch.float32, device=vis.device)
            parts.append(ood)

        f_fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        f_out = self.fusion_mlp(f_fused)
        return self.head(f_out)

    def predict_rank(self, logits: torch.Tensor) -> torch.Tensor:
        return self.head.predict_rank(logits)

    def predict_class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return self.head.predict_class_probs(logits)


def compute_coral_loss(logits: torch.Tensor, targets: torch.Tensor,
                       class_weights: torch.Tensor = None, num_classes: int = 5) -> torch.Tensor:
    """
    Rank-consistent binary cross-entropy loss with optional class weighting.
    """
    num_tasks = num_classes - 1
    levels = torch.arange(num_tasks, device=targets.device).unsqueeze(0)
    binary_targets = (targets.unsqueeze(1) > levels).float()
    bce = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction="none")  # [B, K-1]
    per_sample_loss = bce.mean(dim=1)  # [B]

    if class_weights is not None:
        weights = class_weights[targets]
        loss = (per_sample_loss * weights).sum() / weights.sum().clamp(min=1e-8)
    else:
        loss = per_sample_loss.mean()

    return loss


def compute_effective_num_weights(train_targets: torch.Tensor, beta: float = 0.999,
                                  num_classes: int = 5) -> torch.Tensor:
    """
    Effective Number of Samples Class Weighting (Cui et al., CVPR 2019).
    E_n = (1 - beta^n) / (1 - beta)
    weight_k = 1 / E_{n_k}
    Normalized so sum(weights) = num_classes.
    """
    counts = torch.bincount(train_targets, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    eff_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
    weights = 1.0 / eff_num
    normalized_weights = weights * (num_classes / weights.sum())
    return normalized_weights
