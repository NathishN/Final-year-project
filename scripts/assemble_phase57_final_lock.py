"""
scripts/assemble_phase57_final_lock.py
Assembles the complete Phase 5.7 finalization package in outputs/phase57_final_submission_lock/
and updates outputs/final_submission/.
Computes SHA-256 hashes of all key submission artifacts.
"""

import os
import shutil
import hashlib
import json
import pandas as pd

SRC_DIR = "outputs/final_submission"
LOCK_DIR = "outputs/phase57_final_submission_lock"

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("Assembling Phase 5.7 Final Submission Lock Package...")
    os.makedirs(LOCK_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOCK_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(LOCK_DIR, "figures"), exist_ok=True)
    
    # Copy all files from outputs/final_submission
    for root, dirs, files in os.walk(SRC_DIR):
        rel_path = os.path.relpath(root, SRC_DIR)
        dest_dir = os.path.join(LOCK_DIR, rel_path) if rel_path != "." else LOCK_DIR
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            src_f = os.path.join(root, f)
            dest_f = os.path.join(dest_dir, f)
            shutil.copy2(src_f, dest_f)
            
    # Also ensure gap-specific tables are copied
    gap_tables = [
        "gap1_per_seed_metric_verification.csv",
        "gap1_aggregate_metrics_summary.csv",
        "gap3_statistical_contrasts_verified.csv"
    ]
    for gt in gap_tables:
        src = os.path.join("outputs/final_submission/tables", gt)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(LOCK_DIR, "tables", gt))
            
    print(f"Copied all files to {LOCK_DIR}")
    
    # Compute SHA-256 for key artifacts
    key_files = [
        "manuscript.md",
        "abstract.md",
        "final_results.json",
        "references.bib",
        "CHANGELOG.md",
        "claim_ledger.csv",
        "tables/authoritative_statistical_table.csv",
        "tables/gap1_per_seed_metric_verification.csv",
        "tables/gap1_aggregate_metrics_summary.csv",
        "tables/gap3_statistical_contrasts_verified.csv",
        "tables/ood_metrics_recomputed.csv",
        "tables/split_integrity_summary.csv",
        "tables/calibration_recomputed.csv",
        "tables/selective_prediction_recomputed.csv",
        "tables/figure_manifest.csv",
        "figures/figure1_dataset_composition.png",
        "figures/figure2_qwk_distribution.png",
        "figures/figure3_paired_contrasts.png",
        "figures/figure4_ood_domain_inversion.png",
        "figures/figure5_selective_risk_coverage.png",
        "figures/figure6_inference_benchmarks.png"
    ]
    
    hashes = []
    for rel_f in key_files:
        full_p = os.path.join(LOCK_DIR, rel_f)
        if os.path.exists(full_p):
            h = sha256_file(full_p)
            size = os.path.getsize(full_p)
            hashes.append({
                "artifact": rel_f,
                "size_bytes": size,
                "sha256": h
            })
            print(f"{rel_f:45s} | {size:8d} B | {h}")
        else:
            print(f"WARNING: Missing {rel_f}")
            
    hash_df = pd.DataFrame(hashes)
    hash_csv = os.path.join(LOCK_DIR, "tables", "artifact_sha256_manifest.csv")
    hash_df.to_csv(hash_csv, index=False)
    # Also save in final_submission/tables
    hash_df.to_csv(os.path.join(SRC_DIR, "tables", "artifact_sha256_manifest.csv"), index=False)
    print(f"Saved artifact hash manifest to {hash_csv}")

if __name__ == "__main__":
    main()
