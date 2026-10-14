"""
OOD AUROC recomputation — Phase 5.1 Forensic Audit Task G supplement.
The ood_predictions.csv only has OOD samples (is_ood=1).
We need ID scores from the confirmatory experiment's OOD feature file or
the phase50_ood_benchmark.csv to combine with OOD scores.
"""
import os, json
import pandas as pd
import numpy as np

ROOT = r"C:\fruit_dataset"
OOD_DIR = os.path.join(ROOT, "outputs", "casm_v2", "phase50_ood_features")
FINAL_AUDIT = os.path.join(ROOT, "outputs", "final_audit")
CASM_DIR = os.path.join(ROOT, "outputs", "casm_v2")
AUDIT_OUT = os.path.join(ROOT, "outputs", "phase51_forensic_audit")

# Check benchmark CSV
bench_csv = os.path.join(CASM_DIR, "phase50_ood_benchmark.csv")
print("=== phase50_ood_benchmark.csv ===")
if os.path.isfile(bench_csv):
    df = pd.read_csv(bench_csv)
    print(df.to_string())
else:
    print("MISSING")

# Check features metadata
meta_path = os.path.join(OOD_DIR, "features_metadata.json")
if os.path.isfile(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    print("\n=== features_metadata.json ===")
    print(json.dumps(meta, indent=2))

# Load OOD feature file to check what's in it
import torch
feat_path = os.path.join(OOD_DIR, "ood_real_features.pt")
print("\n=== ood_real_features.pt contents ===")
if os.path.isfile(feat_path):
    data = torch.load(feat_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")
    else:
        print(f"  Type: {type(data)}, shape: {getattr(data, 'shape', 'N/A')}")

# Check if there are combined ID+OOD score files anywhere
print("\n=== Searching for combined score files ===")
for dirpath, _, files in os.walk(os.path.join(ROOT, "outputs")):
    for fn in files:
        if "ood" in fn.lower() and fn.endswith((".csv", ".json", ".pt")):
            fp = os.path.join(dirpath, fn)
            print(f"  {fp} ({os.path.getsize(fp)//1024:.0f} KB)")
