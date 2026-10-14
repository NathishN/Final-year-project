import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from context_audit_train import AuditModel, CORALHead, MODEL_CONFIGS
from severity_model import ContextAwareSeverityNetwork, CORALHead as SeverityCORALHead

print("=== Task 3.3: Reconciling Parameter Counts ===")

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable

configs = {
    "A": {"name": "Image Only", "use_species": False, "use_ood": False, "use_grade": False, "use_unc": False},
    "B": {"name": "Image + Species", "use_species": True, "use_ood": False, "use_grade": False, "use_unc": False},
    "C": {"name": "Image + OOD", "use_species": False, "use_ood": True, "use_grade": False, "use_unc": False},
    "D": {"name": "Image + Species + OOD", "use_species": True, "use_ood": True, "use_grade": False, "use_unc": False},
}

param_audit = {}
for k, cfg in configs.items():
    model = AuditModel(cfg)
    tr, ntr = count_params(model)
    # Break down components
    vis_params = sum(p.numel() for p in model.vis_proj.parameters())
    fc_params = sum(p.numel() for p in model.fc.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    emb_params = sum(p.numel() for p in model.species_embed.parameters()) if cfg["use_species"] else 0
    param_audit[k] = {
        "name": cfg["name"],
        "trainable": tr,
        "non_trainable": ntr,
        "total": tr + ntr,
        "vis_proj": vis_params,
        "species_emb": emb_params,
        "fc_fusion": fc_params,
        "coral_head": head_params
    }
    print(f"Config {k} ({cfg['name']}): Total={tr+ntr:,} | vis_proj={vis_params:,} | species_emb={emb_params} | fc={fc_params:,} | head={head_params}")

print("\n--- Phase 4.0 ContextAwareSeverityNetwork Parameter Counts ---")
for mode in ["A", "B", "C", "D", "E"]:
    casm = ContextAwareSeverityNetwork(ablation_mode=mode)
    tr, ntr = count_params(casm)
    print(f"CASM v1 Mode {mode}: Total={tr+ntr:,}")

# Check Backbone parameter count
print("\n--- ConvNeXt-Tiny Backbone Parameter Count ---")
try:
    import timm
    bb = timm.create_model("convnext_tiny", pretrained=False, num_classes=5)
    tr, ntr = count_params(bb)
    print(f"ConvNeXt-Tiny (full with 5-class head): Total={tr+ntr:,} (Trainable: {tr:,})")
except Exception as e:
    print(f"Could not instantiate timm convnext_tiny: {e}")

print("\n=== Task 3.1: CORAL Mathematical Properties & Unit Tests ===")
# Inspect trained biases from existing Phase 4.3 checkpoints
import glob
ckpts = glob.glob("outputs/phase4/audit/model_*.pth")
print(f"Inspecting biases across {len(ckpts)} Phase 4.3 saved checkpoints...")
bias_violations = []
for cp in ckpts:
    st = torch.load(cp, map_location="cpu", weights_only=False)["state_dict"]
    b = st["head.bias"].numpy()
    # Check monotonicity: b[0] >= b[1] >= b[2] >= b[3]
    diffs = b[:-1] - b[1:]
    is_monotonic = np.all(diffs >= 0)
    if not is_monotonic:
        bias_violations.append((cp, b, diffs))

print(f"Checkpoints with non-monotonic biases: {len(bias_violations)} / {len(ckpts)}")
if bias_violations:
    print("Example non-monotonic bias checkpoint:", bias_violations[0][0])
    print("  Biases:", bias_violations[0][1])
    print("  Diffs (b_k - b_{k+1}):", bias_violations[0][2])

print("\n=== Task 3.2: Species Encoding & Fallback Audit ===")
m_b = AuditModel(configs["B"])
print("Testing species embedding behavior with various inputs:")
# Valid inputs 0..4
spec_valid = torch.tensor([0, 1, 2, 3, 4])
emb_valid = m_b.species_embed(spec_valid)
print("  Valid IDs [0..4] shape:", emb_valid.shape)

# Negative input -1
try:
    spec_neg = torch.tensor([-1])
    emb_neg = m_b.species_embed(spec_neg)
    print("  Negative ID [-1] output:", emb_neg.shape, "-> Matches ID 4?:", torch.equal(emb_neg[0], emb_valid[4]))
except Exception as e:
    print("  Negative ID error:", e)

# Out-of-bounds input 5
try:
    spec_oob = torch.tensor([5])
    emb_oob = m_b.species_embed(spec_oob)
    print("  OOB ID [5] output:", emb_oob.shape)
except Exception as e:
    print("  OOB ID [5] error:", type(e).__name__, str(e))
