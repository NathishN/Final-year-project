"""
severity_model.py
=================
Multi-Input Context-Aware Severity Network (CASM) for Ordinal Multi-Fruit Quality Grading.

Paper Title:
"Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading Using Context-Aware Severity Modeling and Uncertainty Calibration"

Architecture Overview:
Branch A: Visual Features (ConvNeXt-Tiny 768-D bottleneck vector)
Branch B: Fruit Species Categorical Embedding (16-D)
Branch C: Phase 3 Grade Prediction Vector & Grade Confidence (4-D)
Branch D: Open-Set Context (Mahalanobis OOD distance & Unknown Fruit Probability) (2-D)
Branch E: Uncertainty & Confidence (Closed-set Softmax Confidence & Prediction Entropy) (2-D)

Fusion Layer: Gated Cross-Attention / FiLM Context Modulated Fusion Layer
Severity Head: CORAL / CORN / Multi-Class Linear Heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── CORAL Layer Definition ──────────────────────────────────────────────────────

class CORALHead(nn.Module):
    """
    CORAL Output Layer (Cao et al., 2020).
    Uses a single shared weight vector W in R^{1 x D} and K-1 unconstrained biases b_k.
    logits_k = x W^T + b_k  for k = 0, 1, ..., K-2.
    """
    def __init__(self, in_features: int = 128, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1

        self.weight = nn.Parameter(torch.Tensor(1, in_features))
        self.bias = nn.Parameter(torch.Tensor(self.num_tasks))

        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        # weight: (1, D) -> x @ W^T: (B, 1)
        # output: (B, K-1)
        logits_base = torch.matmul(x, self.weight.t())  # (B, 1)
        logits = logits_base + self.bias                # (B, K-1)
        return logits

    def predict_rank(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Converts (B, K-1) logits into predicted ordinal ranks in {0, ..., K-1}.
        rank = count(sigmoid(logits_k) > 0.5)
        """
        probs = torch.sigmoid(logits)
        ranks = (probs > 0.5).sum(dim=1)
        return ranks


# ─── Gated Attention & FiLM Context Fusion Module ───────────────────────────────

class GatedContextFusion(nn.Module):
    """
    Context-Aware Gated Fusion Module.
    Combines visual embeddings with contextual representations via FiLM modulation
    and gated attention weighting.
    """
    def __init__(self, vis_dim: int = 256, ctx_dim: int = 256, fused_dim: int = 256):
        super().__init__()
        # FiLM scale and shift parameters generator from context
        self.film_gen = nn.Linear(ctx_dim, vis_dim * 2)

        # Gated attention gate generator
        self.gate_gen = nn.Sequential(
            nn.Linear(vis_dim + ctx_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, vis_dim),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(vis_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU()
        )

    def forward(self, f_vis: torch.Tensor, f_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_vis: (B, vis_dim) projected visual embeddings.
            f_ctx: (B, ctx_dim) encoded context embeddings.
        Returns:
            f_fused: (B, fused_dim) fused context-aware feature representation.
        """
        # 1. FiLM Modulation: scale (gamma) and shift (beta)
        film_params = self.film_gen(f_ctx)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        f_vis_modulated = (1.0 + gamma) * f_vis + beta

        # 2. Gated Cross-Attention Weight
        concat_feat = torch.cat([f_vis, f_ctx], dim=1)
        alpha = self.gate_gen(concat_feat)  # (B, vis_dim) gate in [0, 1]

        # 3. Dynamic Combination
        f_fused_raw = alpha * f_vis_modulated + (1.0 - alpha) * f_ctx
        f_fused = self.out_proj(f_fused_raw)

        return f_fused


# ─── Main Multi-Input Context-Aware Severity Network (CASM) ─────────────────────

class ContextAwareSeverityNetwork(nn.Module):
    """
    CASM Model Architecture for Phase 4 Ordinal Severity Learning.
    Supports full context fusion as well as context ablation modes ('A', 'B', 'C', 'D', 'E').
    """
    def __init__(
        self,
        vis_in_dim: int = 768,
        num_species: int = 5,
        num_severity_classes: int = 5,
        head_type: str = "coral",  # "coral", "corn", "ce", "focal"
        ablation_mode: str = "E"   # A: Img, B: Img+Spec, C: Img+Spec+Grd, D: Img+Spec+Grd+OOD, E: Full
    ):
        super().__init__()
        self.vis_in_dim = vis_in_dim
        self.num_species = num_species
        self.num_classes = num_severity_classes
        self.head_type = head_type.lower()
        self.ablation_mode = ablation_mode.upper()

        # Visual Branch Projection
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        # Context Embedding & Input Dimensions
        self.species_embed = nn.Embedding(num_species, 16)
        
        # Determine total context input vector size based on ablation mode
        # Mode A: 0 (Visual Only)
        # Mode B: 16 (Species only)
        # Mode C: 16 + 4 (Species + Grade probs/conf) = 20
        # Mode D: 20 + 2 (Grade + OOD distance/prob) = 22
        # Mode E: 22 + 2 (OOD + Uncertainty entropy/conf) = 24
        ctx_raw_dim = self._get_context_dim(self.ablation_mode)

        if ctx_raw_dim > 0:
            self.ctx_mlp = nn.Sequential(
                nn.Linear(ctx_raw_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 256),
                nn.LayerNorm(256),
                nn.GELU()
            )
            self.fusion_module = GatedContextFusion(vis_dim=256, ctx_dim=256, fused_dim=256)
        else:
            self.ctx_mlp = None
            self.fusion_module = None

        # Feature Bottleneck FC Head
        self.fc_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Output Severity Head
        if self.head_type == "coral":
            self.severity_head = CORALHead(in_features=128, num_classes=num_severity_classes)
        elif self.head_type == "corn":
            self.severity_head = nn.Linear(128, num_severity_classes - 1)
        elif self.head_type in ["ce", "focal"]:
            self.severity_head = nn.Linear(128, num_severity_classes)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def _get_context_dim(self, mode: str) -> int:
        if mode == "A":
            return 0
        elif mode == "B":
            return 16
        elif mode == "C":
            return 16 + 4
        elif mode == "D":
            return 16 + 4 + 2
        elif mode == "E":
            return 16 + 4 + 2 + 2
        else:
            raise ValueError(f"Invalid ablation mode: {mode}")

    def build_context_vector(
        self,
        species_id: torch.Tensor,
        grade_ctx: torch.Tensor = None,
        ood_ctx: torch.Tensor = None,
        unc_ctx: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Constructs context vector according to ablation mode.
        """
        if self.ablation_mode == "A":
            return None

        # Branch B: Species Embedding (B, 16)
        spec_vec = self.species_embed(species_id)
        parts = [spec_vec]

        if self.ablation_mode in ["C", "D", "E"]:
            # Branch C: Grade Context (B, 4) -> [p_g0, p_g1, p_g2, max_p_g]
            assert grade_ctx is not None, "grade_ctx required for mode >= C"
            parts.append(grade_ctx)

        if self.ablation_mode in ["D", "E"]:
            # Branch D: OOD Context (B, 2) -> [mah_dist, p_ood]
            assert ood_ctx is not None, "ood_ctx required for mode >= D"
            parts.append(ood_ctx)

        if self.ablation_mode == "E":
            # Branch E: Uncertainty Context (B, 2) -> [entropy, max_conf]
            assert unc_ctx is not None, "unc_ctx required for mode E"
            parts.append(unc_ctx)

        return torch.cat(parts, dim=1)

    def forward(
        self,
        vis_feat: torch.Tensor,
        species_id: torch.Tensor = None,
        grade_ctx: torch.Tensor = None,
        ood_ctx: torch.Tensor = None,
        unc_ctx: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            vis_feat: (B, 768) ConvNeXt-Tiny visual feature vector.
            species_id: (B,) long tensor of species index in {0,1,2,3,4}.
            grade_ctx: (B, 4) grade context.
            ood_ctx: (B, 2) OOD context.
            unc_ctx: (B, 2) uncertainty context.
        Returns:
            logits: Output logits according to head_type.
        """
        # 1. Visual Feature Projection
        f_vis = self.vis_proj(vis_feat)  # (B, 256)

        # 2. Context Processing
        if self.ablation_mode != "A" and species_id is not None:
            raw_ctx = self.build_context_vector(species_id, grade_ctx, ood_ctx, unc_ctx)
            f_ctx = self.ctx_mlp(raw_ctx)  # (B, 256)
            f_fused = self.fusion_module(f_vis, f_ctx)  # (B, 256)
        else:
            f_fused = f_vis  # (B, 256)

        # 3. Feature Bottleneck
        feat_bottleneck = self.fc_head(f_fused)  # (B, 128)

        # 4. Severity Prediction Logits
        logits = self.severity_head(feat_bottleneck)
        return logits

    def predict_rank(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Converts logits into scalar rank predictions in {0, 1, 2, 3, 4}.
        """
        if self.head_type == "coral":
            return self.severity_head.predict_rank(logits)
        elif self.head_type == "corn":
            probs = torch.sigmoid(logits)
            return (probs > 0.5).sum(dim=1)
        elif self.head_type in ["ce", "focal"]:
            return torch.argmax(logits, dim=1)
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")
