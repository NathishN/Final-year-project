"""
casm_v2_corrected_modules.py
============================
Phase 4.5 Corrected Architectural Modules for CASM v2:
1. MonotonicCORALHead: Structurally guarantees monotonic cumulative logits,
   non-negative class probabilities summing to 1.0, and valid ordinal rank decoding.
2. RobustSpeciesEmbedding: Implements 5 known fruit species + index 5 as explicitly
   reserved UNKNOWN_SPECIES fallback, with strict input validation for negative and out-of-range IDs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicCORALHead(nn.Module):
    """
    Structurally Monotonic CORAL Output Layer.
    
    Mathematical Formulation:
        logits_base = x @ W^T  (shape: [B, 1])
        b_0 = theta_0
        b_k = b_{k-1} - softplus(delta_k)   for k = 1, ..., K-2
        
    Because softplus(x) > 0 for all real x:
        b_0 > b_1 > b_2 > ... > b_{K-2}
        
    Therefore:
        logits_{k-1} > logits_k  for all x.
        P(Y > k-1) > P(Y > k)
        
    This mathematically guarantees:
        1. Non-negative class probabilities: P(Y = k) = P(Y > k-1) - P(Y > k) >= 0
        2. Exact probability summation: sum_{k=0}^{K-1} P(Y = k) == 1.0
        3. Monotonic rank prediction without threshold inversion artifacts.
    """
    def __init__(self, in_features: int = 128, num_classes: int = 5):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1  # K-1 binary ordinal tasks

        # Shared projection vector
        self.weight = nn.Parameter(torch.Tensor(1, in_features))
        
        # Base threshold theta_0
        self.theta_0 = nn.Parameter(torch.zeros(1))
        
        # Positive decrements delta_k for k = 1..num_tasks-1
        # Initialized such that softplus(delta) starts at ~1.0
        if self.num_tasks > 1:
            self.raw_deltas = nn.Parameter(torch.ones(self.num_tasks - 1) * 0.5413) # softplus(0.5413) ~= 1.0
        else:
            self.raw_deltas = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def get_biases(self) -> torch.Tensor:
        """Returns the strictly decreasing biases [b_0, b_1, ..., b_{K-2}]."""
        if self.num_tasks == 1:
            return self.theta_0
        deltas = F.softplus(self.raw_deltas)
        # Cumulative subtraction: b_k = theta_0 - sum_{j=1}^k delta_j
        cum_deltas = torch.cumsum(deltas, dim=0)
        tail_biases = self.theta_0 - cum_deltas
        biases = torch.cat([self.theta_0, tail_biases])
        return biases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes ordinal logits: logits_k = x @ W^T + b_k
        Shape: [B, K-1]
        """
        logits_base = torch.matmul(x, self.weight.t())  # [B, 1]
        biases = self.get_biases()                      # [K-1]
        return logits_base + biases

    def predict_cumulative_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """P(Y > k) = sigmoid(logits_k). Guaranteed non-increasing across columns."""
        return torch.sigmoid(logits)

    def predict_class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Converts cumulative probabilities to discrete class distribution P(Y = k).
        Shape: [B, K]
        Guaranteed: P(Y = k) >= 0 and sum(P) == 1.0.
        """
        cum_probs = self.predict_cumulative_probs(logits)  # [B, K-1]
        B = logits.shape[0]
        
        # P(Y = 0) = 1 - P(Y > 0)
        p0 = (1.0 - cum_probs[:, 0:1]).clamp(min=0.0, max=1.0)
        
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        if self.num_tasks > 1:
            pk = (cum_probs[:, :-1] - cum_probs[:, 1:]).clamp(min=0.0, max=1.0)
        else:
            pk = torch.empty((B, 0), device=logits.device)
            
        # P(Y = K-1) = P(Y > K-2)
        p_last = cum_probs[:, -1:].clamp(min=0.0, max=1.0)
        
        class_probs = torch.cat([p0, pk, p_last], dim=1)
        # Normalize to 1.0 for exact numerical stability
        return class_probs / class_probs.sum(dim=1, keepdim=True)

    def predict_rank(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Ordinal rank decoding: count of binary thresholds where P(Y > k) > 0.5.
        Shape: [B]
        """
        cum_probs = self.predict_cumulative_probs(logits)
        return (cum_probs > 0.5).sum(dim=1)


class RobustSpeciesEmbedding(nn.Module):
    """
    Species Embedding module with Open-Set Unknown Species Fallback.
    
    Known species IDs:
        0: Apple
        1: Banana
        2: Mango
        3: Orange
        4: Tomato
        5: UNKNOWN_SPECIES (explicitly reserved open-set fallback)
        
    Properties:
        - Out-of-range, negative, and invalid IDs are safely routed to index 5.
        - Robust under single-image and batched inference.
    """
    UNKNOWN_INDEX = 5
    KNOWN_SPECIES = ["Apple", "Banana", "Mango", "Orange", "Tomato"]

    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        self.num_categories = 6  # 5 known + 1 unknown
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_categories, embedding_dim)
        
        # Xavier/Kaiming initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def sanitize_ids(self, species_ids: torch.Tensor) -> torch.Tensor:
        """
        Validates and routes out-of-range or negative indices to UNKNOWN_INDEX (5).
        """
        if not torch.is_tensor(species_ids):
            species_ids = torch.as_tensor(species_ids, dtype=torch.long)
        else:
            species_ids = species_ids.long()

        # Mask for valid known IDs: 0 <= id <= 4
        is_known = (species_ids >= 0) & (species_ids < 5)
        
        # Safe IDs: replace any unknown / negative / >=5 with UNKNOWN_INDEX (5)
        safe_ids = torch.where(is_known, species_ids, torch.full_like(species_ids, self.UNKNOWN_INDEX))
        return safe_ids

    def forward(self, species_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic ID sanitization.
        Returns tensor of shape [B, embedding_dim].
        """
        safe_ids = self.sanitize_ids(species_ids).to(self.embedding.weight.device)
        return self.embedding(safe_ids)
