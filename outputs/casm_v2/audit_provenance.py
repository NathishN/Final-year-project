import os
import hashlib
import json
import numpy as np
import torch
import pandas as pd

def get_file_info(filepath):
    if not os.path.exists(filepath):
        return {"exists": False, "size_bytes": 0, "sha256": "NOT_FOUND"}
    size = os.path.getsize(filepath)
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 1MB chunks
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return {"exists": True, "size_bytes": size, "sha256": hasher.hexdigest()[:16]}

print("Checking key artifact files...")
artifacts = [
    "outputs/features/phase4_dataset.pt",
    "outputs/checkpoints/best_model.pth",
    "outputs/coral/checkpoints/coral_best_model.pth",
    "master_dataset.csv",
    "severity_mapping.json",
    "casmv2_validation_results.csv",
    "outputs/casm_v2/phase44_ablation_raw.csv",
    "outputs/casm_v2/phase44_ablation_results.csv",
    "outputs/casm_v2/phase44_ablation_significance.csv",
    "outputs/casm_v2/phase44_recalculated_metrics.csv",
    "outputs/casm_v2/phase44_per_seed_differences.csv",
    "outputs/casm_v2/phase44_statistical_report_data.csv",
    "outputs/casm_v2/phase44_bootstrap_data.json",
]

for art in artifacts:
    info = get_file_info(art)
    print(f"{art}: exists={info['exists']}, size={info['size_bytes']}, sha256_short={info['sha256']}")

# Check phase4_dataset.pt structure
feat_path = "outputs/features/phase4_dataset.pt"
if os.path.exists(feat_path):
    print("\nInspecting outputs/features/phase4_dataset.pt...")
    data = torch.load(feat_path, map_location="cpu", weights_only=False)
    print(f"Keys: {list(data.keys())}")
    for k in data.keys():
        val = data[k]
        if isinstance(val, torch.Tensor):
            print(f"  {k}: tensor {val.shape}, dtype={val.dtype}")
        elif isinstance(val, np.ndarray):
            print(f"  {k}: ndarray {val.shape}, dtype={val.dtype}")
        else:
            print(f"  {k}: type={type(val)}")
            if hasattr(val, '__len__'):
                print(f"    length={len(val)}")

# Check outputs/checkpoints/best_model.pth
ckpt_path = "outputs/checkpoints/best_model.pth"
if os.path.exists(ckpt_path):
    print("\nInspecting outputs/checkpoints/best_model.pth...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    if isinstance(ckpt, dict):
        for k in ["epoch", "val_f1", "val_acc", "accuracy", "f1"]:
            if k in ckpt:
                print(f"  {k}: {ckpt[k]}")

# Check outputs/phase4/ablation checkpoints
ablation_dir = "outputs/phase4/ablation"
if os.path.exists(ablation_dir):
    print(f"\nFiles in {ablation_dir}:", os.listdir(ablation_dir))

# Check outputs/phase4/audit checkpoints
audit_dir = "outputs/phase4/audit"
if os.path.exists(audit_dir):
    print(f"Checkpoints in {audit_dir}: count={len(os.listdir(audit_dir))}")

# Check outputs/casm_v2/ablation_ckpts
v2_ckpts = "outputs/casm_v2/ablation_ckpts"
if os.path.exists(v2_ckpts):
    print(f"Files in {v2_ckpts}:", os.listdir(v2_ckpts))
