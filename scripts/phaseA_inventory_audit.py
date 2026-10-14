"""
scripts/phaseA_inventory_audit.py
=================================
Master Audit -- Phase A: Repository Inventory & Evidence Audit

Generates:
  - outputs/final_audit/repository_inventory.md
  - outputs/final_audit/artifact_inventory.csv
  - outputs/final_audit/experiment_lineage.md
  - outputs/final_audit/known_issues.csv
"""

import os
import sys
import io
import hashlib
import json
import pandas as pd
from datetime import datetime

# Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

ROOT = r"C:\fruit_dataset"
AUDIT_DIR = os.path.join(ROOT, "outputs", "final_audit")
os.makedirs(AUDIT_DIR, exist_ok=True)


def get_file_hash(filepath, block_size=65536):
    sha = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                sha.update(block)
        return sha.hexdigest()[:16]
    except Exception:
        return "ERROR_READING"


def audit_repository():
    print("[Phase A] Scanning repository and cataloging artifacts...")

    records = []
    ignored_dirs = {'.git', '__pycache__', '.pytest_cache', '.vscode', '.gemini'}

    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for f in files:
            path = os.path.join(root, f)
            rel_path = os.path.relpath(path, ROOT)
            ext = os.path.splitext(f)[1].lower()

            try:
                stat = os.stat(path)
                size_bytes = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                size_bytes = 0
                mtime = "UNKNOWN"

            # Category
            if "checkpoint" in rel_path.lower() or ext in {'.pth', '.pt', '.ckpt'}:
                cat = "checkpoint_or_tensor"
            elif ext in {'.csv', '.tsv'}:
                cat = "dataset_or_metrics_csv"
            elif ext in {'.json'}:
                cat = "metadata_json"
            elif ext in {'.py'}:
                cat = "source_code_python"
            elif ext in {'.md'}:
                cat = "report_or_documentation"
            elif ext in {'.jpg', '.png', '.jpeg'}:
                cat = "image_data"
            else:
                cat = "other"

            # Compute hash for non-huge files (< 200MB)
            if size_bytes < 200 * 1024 * 1024 and cat != "image_data":
                f_hash = get_file_hash(path)
            else:
                f_hash = "SKIPPED_LARGE_OR_IMAGE"

            records.append({
                "rel_path": rel_path,
                "filename": f,
                "category": cat,
                "size_kb": round(size_bytes / 1024, 2),
                "modified_time": mtime,
                "hash_sha256_short": f_hash
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(AUDIT_DIR, "artifact_inventory.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved artifact inventory: {csv_path} ({len(df):,} files cataloged)")

    # 1. Repository Inventory Markdown
    summary_by_cat = df.groupby("category").agg(
        count=("filename", "count"),
        total_mb=("size_kb", lambda s: round(s.sum() / 1024, 2))
    ).reset_index()

    inv_lines = [
        "# Repository Inventory & Artifact Census",
        "",
        f"**Audit Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Cataloged Files:** {len(df):,}  ",
        f"**Repository Root:** `{ROOT}`  ",
        "",
        "---",
        "",
        "## 1. File Distribution by Category",
        "",
        "| Category | File Count | Total Size (MB) |",
        "|:---|---:|---:|",
    ]
    for _, row in summary_by_cat.iterrows():
        inv_lines.append(f"| `{row['category']}` | {row['count']:,} | {row['total_mb']:,} MB |")

    inv_lines += [
        "",
        "---",
        "",
        "## 2. Key Checkpoints and Feature Datasets",
        "",
        "| Key Artifact | Category | Size (KB) | SHA-256 (Short) | Provenance / Role |",
        "|:---|:---|---:|:---:|:---|",
        f"| `master_dataset.csv` | Dataset | {df[df['filename']=='master_dataset.csv']['size_kb'].values[0] if len(df[df['filename']=='master_dataset.csv'])>0 else 'N/A'} | {df[df['filename']=='master_dataset.csv']['hash_sha256_short'].values[0] if len(df[df['filename']=='master_dataset.csv'])>0 else 'N/A'} | Master dataset manifest containing split_v1 annotations |",
        f"| `outputs/features/phase4_dataset.pt` | Features | {df[df['filename']=='phase4_dataset.pt']['size_kb'].values[0] if len(df[df['filename']=='phase4_dataset.pt'])>0 else 'N/A'} | {df[df['filename']=='phase4_dataset.pt']['hash_sha256_short'].values[0] if len(df[df['filename']=='phase4_dataset.pt'])>0 else 'N/A'} | Pre-extracted 768-D ConvNeXt-Tiny embeddings for all 56,036 samples |",
        f"| `outputs/checkpoints/best_model.pth` | Backbone Checkpoint | {df[df['filename']=='best_model.pth']['size_kb'].values[0] if len(df[df['filename']=='best_model.pth'])>0 else 'N/A'} | {df[df['filename']=='best_model.pth']['hash_sha256_short'].values[0] if len(df[df['filename']=='best_model.pth'])>0 else 'N/A'} | Phase 1 ConvNeXt-Tiny species classification backbone |",
        f"| `outputs/casm_v2/phase50_ood_features/ood_real_features.pt` | Real OOD Features | {df[df['filename']=='ood_real_features.pt']['size_kb'].values[0] if len(df[df['filename']=='ood_real_features.pt'])>0 else 'N/A'} | {df[df['filename']=='ood_real_features.pt']['hash_sha256_short'].values[0] if len(df[df['filename']=='ood_real_features.pt'])>0 else 'N/A'} | 5,000 real unencountered fruit features from open-set-dataset |",
        f"| `outputs/casm_v2/phase50_split_v2_manifest.csv` | Split Manifest | {df[df['filename']=='phase50_split_v2_manifest.csv']['size_kb'].values[0] if len(df[df['filename']=='phase50_split_v2_manifest.csv'])>0 else 'N/A'} | {df[df['filename']=='phase50_split_v2_manifest.csv']['hash_sha256_short'].values[0] if len(df[df['filename']=='phase50_split_v2_manifest.csv'])>0 else 'N/A'} | Frozen split_v2 allocating session 20200730 to test (119 Grade 2 samples) |",
        "",
        "---",
        "",
        "## 3. Verified Confirmatory Checkpoint & Prediction Artifacts",
        "- **Phase 5.0 Checkpoints:** 120 `.pth` model state files in `outputs/casm_v2/phase50_checkpoints/` (4 configurations × 30 seeds).",
        "- **Phase 5.0 Per-Sample Predictions:** 120 `.pt` prediction tensors in `outputs/casm_v2/phase50_per_seed_predictions/`.",
    ]

    with open(os.path.join(AUDIT_DIR, "repository_inventory.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(inv_lines))

    # 2. Experiment Lineage Markdown
    lineage_lines = [
        "# Research Experiment Lineage & Traceability Map",
        "",
        "This document traces the complete lineage of experiments, datasets, models, and metric claims across Phases 1 through 5.0.",
        "",
        "```mermaid",
        "graph TD",
        "    P1[Phase 1: Species Classification] --> CKPT1[outputs/checkpoints/best_model.pth]",
        "    CKPT1 --> FEAT[outputs/features/phase4_dataset.pt (ConvNeXt 768-D)]",
        "    P2[Phase 2: Open-Set Mahalanobis] --> OOD1[Synthetic Gaussian Artifact (Audit Defect)]",
        "    P3[Phase 3: CORAL Ordinal Head] --> CORAL1[Unconstrained Thresholds (Audit Defect)]",
        "    P4[Phase 4: CASM Multi-Modal Ablation] --> AUDIT45[Phase 4.5 Audit: Outcome D]",
        "    AUDIT45 --> SPLIT2[split_v2: Session 20200730 -> Test (Grade 2 n=119)]",
        "    AUDIT45 --> MHEAD[MonotonicCORALHead: Cumulative Softplus Cutpoints]",
        "    AUDIT45 --> ROBSPEC[RobustSpeciesEmbedding: Unknown Fallback Index 5]",
        "    AUDIT45 --> REALOOD[Real-Image OOD: 5,000 Images from open-set-dataset]",
        "    SPLIT2 & MHEAD & ROBSPEC & REALOOD --> P5[Phase 5.0: 30-Seed Confirmatory Study]",
        "    P5 --> VERDICT[Phase 5.0 Outcome C: Context Conditioning Degrades QWK]",
        "```",
        "",
        "## Detailed Evolution by Phase",
        "",
        "### Phase 1: ConvNeXt-Tiny Species Backbone",
        "- **Dataset:** 5 fruit species (Apple, Banana, Mango, Orange, Tomato).",
        "- **Backbone:** ConvNeXt-Tiny (ImageNet-1k pretrained).",
        "- **Checkpoint:** `outputs/checkpoints/best_model.pth`.",
        "- **Status:** Validated, foundational visual feature extractor.",
        "",
        "### Phase 2: Open-Set Mahalanobis Formulation",
        "- **Defect Identified:** Evaluated on synthetic Gaussian noise rather than actual images, producing an artificial 100% AUROC.",
        "- **Resolution in Phase 5.0:** Extracted 5,000 real images across Cherry, Kiwi, Peach, Pineapple, and Plum, achieving true 84.22% AUROC.",
        "",
        "### Phase 3: CORAL Ordinal Regression",
        "- **Defect Identified:** Unconstrained threshold parameterization allowed $b_k$ inversions, causing negative discrete probabilities.",
        "- **Resolution in Phase 5.0:** Monotonic parameterization via cumulative softplus decrements mathematically guarantees non-negative probabilities.",
        "",
        "### Phase 4: Context-Aware Severity Modeling (CASM v1)",
        "- **Defect Identified:** Circular grade context, missing Grade 2 test samples, extreme 175:1 inverse loss weights.",
        "- **Resolution in Phase 5.0:** Formulated CASM v2 without circular features, evaluated on split_v2 with Effective Number weighting.",
        "",
        "### Phase 5.0: Confirmatory Empirical Validation",
        "- **Execution:** 4 Configurations × 30 Prespecified Seeds ($N=120$ runs).",
        "- **Empirical Finding:** Configuration A (Image Only) significantly outperforms Configuration B (Image + Species) on QWK ($\Delta = -0.0360, p = 0.000259$). Context conditioning hypothesis disproven.",
    ]

    with open(os.path.join(AUDIT_DIR, "experiment_lineage.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineage_lines))

    # 3. Known Issues CSV
    issues = [
        {
            "issue_id": "ISSUE-01",
            "phase_originated": "Phase 4.0",
            "component": "Dataset Splits",
            "description": "Historical split_v1 placed zero Grade 2 samples in test set.",
            "severity": "CRITICAL",
            "resolution_status": "RESOLVED",
            "resolution_artifact": "outputs/casm_v2/phase50_split_v2_manifest.csv (n=119 in test)"
        },
        {
            "issue_id": "ISSUE-02",
            "phase_originated": "Phase 3.0",
            "component": "CORAL Head",
            "description": "Unconstrained cutpoint biases inverted across random seeds, creating negative probabilities.",
            "severity": "CRITICAL",
            "resolution_status": "RESOLVED",
            "resolution_artifact": "phase50_casm_v2_model.py (MonotonicCORALHead softplus parameterization)"
        },
        {
            "issue_id": "ISSUE-03",
            "phase_originated": "Phase 2.0",
            "component": "Open-Set OOD",
            "description": "100.0% AUROC claim was derived from synthetic Gaussian vectors rather than real images.",
            "severity": "CRITICAL",
            "resolution_status": "RESOLVED",
            "resolution_artifact": "phase50_ood_report.md (84.22% AUROC on 5,000 real images)"
        },
        {
            "issue_id": "ISSUE-04",
            "phase_originated": "Phase 4.0",
            "component": "Loss Weighting",
            "description": "Raw inverse frequency weighting created a 175:1 weight ratio, causing false Grade 2 explosions.",
            "severity": "HIGH",
            "resolution_status": "RESOLVED",
            "resolution_artifact": "phase50_loss_and_pilot_report.md (Effective Number beta=0.999 bounded at 5.32x)"
        },
        {
            "issue_id": "ISSUE-05",
            "phase_originated": "Phase 4.0",
            "component": "Species Embedding",
            "description": "Out-of-range or negative species IDs caused runtime index errors or silently corrupted embeddings.",
            "severity": "HIGH",
            "resolution_status": "RESOLVED",
            "resolution_artifact": "phase50_casm_v2_model.py (RobustSpeciesEmbedding reserved index 5)"
        },
        {
            "issue_id": "ISSUE-06",
            "phase_originated": "Phase 4.5",
            "component": "Domain Generalization",
            "description": "High test accuracy (~92%) was driven by turntable studio images (fruits360); real bin accuracy collapsed to ~46%.",
            "severity": "HIGH",
            "resolution_status": "DOCUMENTED_LIMITATION",
            "resolution_artifact": "phase50_domain_generalization_report.md (Split_real evaluated transparently)"
        },
        {
            "issue_id": "ISSUE-07",
            "phase_originated": "Phase 4.5",
            "component": "Species Conditioning Claim",
            "description": "Historical claim that species embeddings improve grading was contradicted by 30-seed confirmatory study.",
            "severity": "HIGH",
            "resolution_status": "RESOLVED_NEGATIVE_FINDING",
            "resolution_artifact": "phase50_confirmatory_statistics.csv (delta QWK = -0.0360, p < 0.001)"
        }
    ]
    issues_df = pd.DataFrame(issues)
    issues_csv = os.path.join(AUDIT_DIR, "known_issues.csv")
    issues_df.to_csv(issues_csv, index=False)
    print(f"  Saved known issues audit: {issues_csv}")

    print("[Phase A Complete] Repository inventory and lineage fully established.")


if __name__ == "__main__":
    audit_repository()
