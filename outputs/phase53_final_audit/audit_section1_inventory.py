"""
outputs/phase53_final_audit/audit_section1_inventory.py
=========================================================
Performs Section 1: Deep Repository & Artifact Inspection
Generates: outputs/phase53_final_audit/INITIAL_INVENTORY.md
"""

import os
import sys
import hashlib
import json
import datetime
import pandas as pd
import torch

ROOT = r"C:\fruit_dataset"
OUT_DIR = os.path.join(ROOT, "outputs", "phase53_final_audit")
os.makedirs(OUT_DIR, exist_ok=True)

def hash_file(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_file_info(rel_path):
    full = os.path.join(ROOT, rel_path)
    exists = os.path.exists(full)
    sz = os.path.getsize(full) if exists else 0
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(full)).isoformat() if exists else None
    return {
        "rel_path": rel_path,
        "full_path": full,
        "exists": exists,
        "size_bytes": sz,
        "mtime": mtime
    }

key_artifacts = [
    # Phase 5.2 Core
    "outputs/phase52_final_validation/FINAL_REPORT.md",
    "outputs/phase52_final_validation/CLAIM_LEDGER.csv",
    "outputs/phase52_final_validation/PROGRESS.md",
    "outputs/phase52_final_validation/COMMAND_LOG.md",
    "outputs/phase52_final_validation/DECISIONS.md",
    "outputs/phase52_final_validation/ISSUES.csv",
    "outputs/phase52_final_validation/run_verification_matrix.csv",
    "outputs/phase52_final_validation/run_verification_report.md",
    "outputs/phase52_final_validation/training_protocol_verification.md",
    "outputs/phase52_final_validation/recomputed_metrics_by_configuration.csv",
    "outputs/phase52_final_validation/recomputed_metrics_by_seed.csv",
    "outputs/phase52_final_validation/metric_discrepancies.csv",
    "outputs/phase52_final_validation/paired_differences.csv",
    "outputs/phase52_final_validation/statistical_reconciliation.md",
    "outputs/phase52_final_validation/split_audit.csv",
    "outputs/phase52_final_validation/split_audit_report.md",
    "outputs/phase52_final_validation/session_holdout_manifest.csv",
    "outputs/phase52_final_validation/session_holdout_results.csv",
    "outputs/phase52_final_validation/session_holdout_report.md",
    "outputs/phase52_final_validation/ood_metrics_recomputed.csv",
    "outputs/phase52_final_validation/ood_species_results.csv",
    "outputs/phase52_final_validation/ood_domain_results.csv",
    "outputs/phase52_final_validation/ood_operating_points.csv",
    "outputs/phase52_final_validation/ood_confounding_report.md",
    "outputs/phase52_final_validation/calibration_recomputed.csv",
    "outputs/phase52_final_validation/selective_prediction_results.csv",
    "outputs/phase52_final_validation/calibration_and_abstention_report.md",
    "outputs/phase52_final_validation/e2e_benchmark_results.csv",
    "outputs/phase52_final_validation/e2e_benchmark_report.md",
    "outputs/phase52_final_validation/reproduction_command_results.csv",
    "outputs/phase52_final_validation/integration_test_audit.md",
    # Session Holdout Retrain (120 runs)
    "outputs/phase52_final_validation/session_holdout_retrain/summary_by_config.csv",
    "outputs/phase52_final_validation/session_holdout_retrain/paired_stats.csv",
    "outputs/phase52_final_validation/session_holdout_retrain/per_seed_results.csv",
    # Checkpoints & Tensors
    "outputs/checkpoints/best_model.pth",
    "outputs/features/phase4_dataset.pt",
    "outputs/casm_v2/phase50_split_v2_manifest.csv",
    "outputs/casm_v2/phase50_ood_features/ood_real_features.pt",
    "outputs/casm_v2/phase50_ood_features/features_metadata.json",
    "outputs/final_audit/ood_predictions.csv",
    # Manuscript
    "paper/manuscript.md",
    "paper/abstract.md",
    "paper/methodology.md",
    "paper/results.md",
    "paper/limitations.md",
    "paper/references.bib",
    "paper/final_reproducibility_checklist.md",
    "paper/final_claim_evidence_matrix.csv",
    "paper/publication_readiness_assessment.md",
    "paper/manuscript_revision_log.md",
    # Tests
    "tests/test_data_integrity.py",
    "tests/test_model_properties.py",
    "tests/test_metrics_and_calibration.py",
    "tests/test_end_to_end_smoke.py",
    "tests/test_integration_predictions.py"
]

inventory_records = []
for p in key_artifacts:
    info = get_file_info(p)
    inventory_records.append(info)

# Check 120 historical checkpoints
hist_ckpt_dir = os.path.join(ROOT, "outputs", "casm_v2", "phase50_checkpoints")
hist_ckpts = [f for f in os.listdir(hist_ckpt_dir) if f.endswith(".pth")] if os.path.exists(hist_ckpt_dir) else []
# Check 120 historical predictions
hist_pred_dir = os.path.join(ROOT, "outputs", "casm_v2", "phase50_per_seed_predictions")
hist_preds = [f for f in os.listdir(hist_pred_dir) if f.endswith(".pt")] if os.path.exists(hist_pred_dir) else []

# Check 120 holdout retrain checkpoints
retrain_ckpt_dir = os.path.join(ROOT, "outputs", "phase52_final_validation", "session_holdout_retrain", "checkpoints")
retrain_ckpts = [f for f in os.listdir(retrain_ckpt_dir) if f.endswith(".pth")] if os.path.exists(retrain_ckpt_dir) else []
# Check 120 holdout retrain predictions
retrain_pred_dir = os.path.join(ROOT, "outputs", "phase52_final_validation", "session_holdout_retrain", "predictions")
retrain_preds = [f for f in os.listdir(retrain_pred_dir) if f.endswith(".pt")] if os.path.exists(retrain_pred_dir) else []

# Assemble INITIAL_INVENTORY.md
with open(os.path.join(OUT_DIR, "INITIAL_INVENTORY.md"), "w", encoding="utf-8") as f:
    f.write("# Phase 5.3 — Initial Repository and Artifact Inventory\n\n")
    f.write(f"**Date of Audit:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    f.write(f"**Auditor:** Senior Forensic Reviewer, Computer Vision & Statistics  \n")
    f.write(f"**Repository Root:** `{ROOT}`  \n\n")
    f.write("---\n\n")
    f.write("## 1. Executive Summary of Available Evidence\n\n")
    f.write(f"- **Key Tracked Artifacts:** {sum(1 for r in inventory_records if r['exists'])} / {len(inventory_records)} present on disk.\n")
    f.write(f"- **Historical Confirmatory Checkpoints:** {len(hist_ckpts)} / 120 present in `outputs/casm_v2/phase50_checkpoints/`.\n")
    f.write(f"- **Historical Confirmatory Predictions:** {len(hist_preds)} / 120 present in `outputs/casm_v2/phase50_per_seed_predictions/`.\n")
    f.write(f"- **Session-Held-Out Retrained Checkpoints:** {len(retrain_ckpts)} / 120 present in `outputs/phase52_final_validation/session_holdout_retrain/checkpoints/`.\n")
    f.write(f"- **Session-Held-Out Retrained Predictions:** {len(retrain_preds)} / 120 present in `outputs/phase52_final_validation/session_holdout_retrain/predictions/`.\n")
    f.write(f"- **Master Dataset Feature Cache:** `outputs/features/phase4_dataset.pt` (179.2 MB, 56,036 samples, 768-D).\n")
    f.write(f"- **Real OOD Feature Cache:** `outputs/casm_v2/phase50_ood_features/ood_real_features.pt` (15.9 MB, 5,000 samples across 5 unseen species).\n\n")
    
    f.write("## 2. Inventory of Critical Files\n\n")
    f.write("| Relative Path | Exists | Size (Bytes) | Last Modified |\n")
    f.write("|---|:---:|---:|:---|\n")
    for r in inventory_records:
        status_str = "YES" if r["exists"] else "**MISSING**"
        f.write(f"| `{r['rel_path']}` | {status_str} | {r['size_bytes']:,} | {r['mtime']} |\n")
        
    f.write("\n## 3. Scripts and Corresponding Regeneration Commands\n\n")
    f.write("| Result Domain | Source Script | Input Artifacts | Output Artifacts | Exact Command |\n")
    f.write("|---|---|---|---|---|\n")
    f.write("| Unit & Integration Tests | `tests/` | Prediction tensors, models | Test execution log | `python -m unittest discover tests -v` |\n")
    f.write("| Split & Dataset Integrity | `scripts/phaseB_dataset_integrity.py` | `outputs/casm_v2/phase50_split_v2_manifest.csv` | `outputs/final_audit/split_integrity.csv` | `python scripts/phaseB_dataset_integrity.py` |\n")
    f.write("| Ordinal Loss & Cutpoints | `scripts/phaseC_ordinal_loss_audit.py` | `phase50_casm_v2_model.py` | Validation logs | `python scripts/phaseC_ordinal_loss_audit.py` |\n")
    f.write("| Hardware Latency & E2E | `scripts/phaseH_model_efficiency.py` | `outputs/checkpoints/best_model.pth` | `outputs/final_audit/latency_benchmarks.csv` | `python scripts/phaseH_model_efficiency.py` |\n")
    f.write("| Manuscript Assembly | `scripts/assemble_paper.py` | `paper/*.md`, `references.bib` | `paper/manuscript.md` | `python scripts/assemble_paper.py` |\n")
    f.write("| 120 Historical Runs | `phase50_stage5_confirmatory.py` | `outputs/features/phase4_dataset.pt` | `outputs/casm_v2/phase50_checkpoints/`, `phase50_per_seed_predictions/` | `python phase50_stage5_confirmatory.py` |\n")
    f.write("| 120 Session Holdout Runs | `outputs/phase52_final_validation/run_session_holdout_retrain.py` | `session_holdout_manifest.csv`, `phase4_dataset.pt` | `outputs/phase52_final_validation/session_holdout_retrain/` | `python outputs/phase52_final_validation/run_session_holdout_retrain.py` |\n\n")

    f.write("## 4. Inconsistencies & High-Risk Questions to Reconcile\n\n")
    f.write("1. **Session-Held-Out Protocol Integrity:** In `split_v2`, Session `20200730` had 119 Grade 2 Banana images in test, but 147 samples in train. Phase 5.2 generated `session_holdout_manifest.csv` (12,980 test samples) and retrained all 120 runs. Must independently audit every session ID, sample count, hash, and leakage vector.\n")
    f.write("2. **Statistical Reconciliation:** B vs A degradation reported as $-0.0360$ (historical) and $-0.0262$ (session-held-out). Wilcoxon $W=72.0$ vs prior citations of 59 and 62 must be verified mathematically with exact zero handling, tie handling, and Holm-Bonferroni correction.\n")
    f.write("3. **Calibration Provenance:** Recomputed ECE of 1.45% (10 bins) vs historical 2.51% and selective top-90% accuracy of 99.28% vs 99.14% must be traced to exact split, checkpoint, and validation-tuning rules. Domain-stratified calibration must expose studio dominance.\n")
    f.write("4. **OOD Domain-Mapping Inconsistency:** Clarify the exact correspondence between `real_quality`, `real_grade`, and conveyor/market descriptors in text and tables. Audit Mahalanobis inversion (18.08%) vs Energy score (92.49% on `real_quality`, 98.89% on `real_grade`). Confirm why causal background attribution is scientifically unresolved.\n")
    f.write("5. **Hardware Benchmark Verification:** Verify end-to-end inference benchmark on RTX 3050 (including image loading, preprocessing, backbone, and head) versus head-only throughput.\n")

print("Generated INITIAL_INVENTORY.md successfully.")
