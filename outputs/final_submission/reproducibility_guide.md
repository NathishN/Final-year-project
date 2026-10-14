# Reproducibility Guide

**Project:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation  
**Target Venue:** IEEE Transactions on Agri-Food Computing / Pattern Analysis & Machine Intelligence  
**Artifact Directory:** `./` (Repository Root)  

---

## 1. Environment Specification

- **OS:** Windows 11 Enterprise (64-bit) / Linux Ubuntu 22.04 LTS
- **Python:** 3.12.4
- **PyTorch:** 2.5.1+cu121 (CUDA 12.1)
- **Key Python Packages:**
  - `torch >= 2.5.0`
  - `scipy >= 1.17.0`
  - `numpy >= 2.0.0`
  - `pandas >= 2.2.0`
  - `scikit-learn >= 1.6.0`
  - `matplotlib >= 3.9.0`
  - `pytest >= 8.3.0`
- **GPU Benchmark Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, 60W TGP)

---

## 2. Directory Structure and Artifact Mapping

```text
fruit_dataset/
├── outputs/
│   ├── casm_v2/
│   │   ├── phase50_per_seed_predictions/      # 120 .pt prediction tensors (Historical Split V2)
│   │   ├── phase50_checkpoints/               # Model checkpoints (.pth)
│   │   ├── phase50_split_v2_manifest.csv      # 56,036 sample manifest (Historical Split V2)
│   │   └── phase50_ood_features/              # Real OOD features (5,000 real images)
│   ├── phase52_final_validation/
│   │   ├── session_holdout_retrain/
│   │   │   ├── predictions/                   # 120 .pt prediction tensors (Session Holdout)
│   │   │   └── per_seed_results.csv           # Seed-level results table
│   │   └── session_holdout_manifest.csv       # 56,036 sample manifest (Session Holdout)
│   └── final_submission/
│       ├── manuscript.md                      # Complete peer-review ready manuscript
│       ├── abstract.md                        # Standalone abstract
│       ├── final_results.json                 # Machine-readable single source of truth
│       ├── claim_ledger.csv                   # Forensic claim audit ledger
│       ├── reproducibility_guide.md           # This document
│       ├── environment_info.txt               # Hardware and dependency lockfile
│       ├── test_results.txt                   # Automated test log (28/28 tests passed)
│       ├── data_and_code_availability.md      # Data and code access policies
│       ├── CHANGELOG.md                       # Comprehensive progression changelog
│       ├── FINAL_SUBMISSION_REPORT.md         # Final certification verdict
│       ├── tables/                            # Authoritative CSV tables
│       └── figures/                           # Publication figures at 300 DPI
├── scripts/
│   ├── generate_authoritative_final_results.py
│   ├── generate_final_submission_figures.py
│   ├── generate_authoritative_claim_ledger.py
│   └── compile_full_manuscript.py
└── tests/
    └── test_phase56_regression_hardening.py   # Regression test suite
```

---

## 3. Step-by-Step Reproduction Commands

All commands are executed from the repository root (`./`):

### Step 1: Run Full Automated Regression Test Suite
Validates the Monotonic CORAL head, softplus biases, split integrity, exact conditional permutation Wilcoxon p-value ($p=0.000514$), Holm multiplicity adjustment, and real checkpoint parameter census:
```bash
pytest -v
```
*Expected Outcome:* 28 passed in ~110 seconds, Exit code 0.

### Step 2: Regenerate Single Source of Truth (`final_results.json`)
Extracts and recomputes all metrics directly from the 240 raw `.pt` prediction tensors across both Historical Split V2 and 100% Group Session Holdout:
```bash
python scripts/generate_authoritative_final_results.py
```
*Outputs Generated:*
- `outputs/final_submission/final_results.json`
- `outputs/final_submission/tables/authoritative_statistical_table.csv`

### Step 3: Recompute Real-Image OOD Benchmarks
Evaluates Mahalanobis distance, Free Energy score, and Maximum Softmax Probability on 5,000 real images of unencountered species (*Cherry, Kiwi, Peach, Pineapple, Plum*):
```bash
python outputs/phase56_publication_lock/revalidate_ood.py
```
*Outputs Generated:*
- `outputs/final_submission/tables/ood_metrics_recomputed.csv`

### Step 4: Verify Split Manifests and Session Isolation
Audits `phase50_split_v2_manifest.csv` and `session_holdout_manifest.csv` across all 56,036 image paths, hashes, and 80 physical acquisition sessions:
```bash
python outputs/phase56_publication_lock/verify_split_integrity.py
```
*Outputs Generated:*
- `outputs/final_submission/tables/split_integrity_summary.csv`

### Step 5: Regenerate All Publication Figures at 300 DPI
Plots Figures 1 through 6 directly from `final_results.json`:
```bash
python scripts/generate_final_submission_figures.py
```
*Outputs Generated:*
- `outputs/final_submission/figures/figure1_dataset_composition.png`
- `outputs/final_submission/figures/figure2_qwk_distribution.png`
- `outputs/final_submission/figures/figure3_paired_contrasts.png`
- `outputs/final_submission/figures/figure4_ood_domain_inversion.png`
- `outputs/final_submission/figures/figure5_selective_risk_coverage.png`
- `outputs/final_submission/figures/figure6_inference_benchmarks.png`
- `outputs/final_submission/tables/figure_manifest.csv`

### Step 6: Compile Full Peer-Review Ready Manuscript
Compiles and synchronizes `paper/manuscript.md` and `outputs/final_submission/manuscript.md`:
```bash
python scripts/compile_full_manuscript.py
```

---

## 4. Key Metrics Quick Reference

| Quantity | Historical Split V2 | 100% Group Session Holdout |
|:---|:---:|:---:|
| Total Dataset Samples | 56,036 | 56,036 |
| Test Set Images ($N_{\text{test}}$) | 12,833 | 12,980 |
| Cross-Split Leaked Sessions | 1 (Session 20200730: 147 train / 119 test) | **0 across all 80 sessions** |
| Paired Training Seeds ($n_{\text{seeds}}$) | 30 | 30 |
| Baseline A QWK | **0.8478 ± 0.0260** | **0.8038 ± 0.0335** |
| Species B QWK | 0.8118 ± 0.0364 | 0.7776 ± 0.0431 |
| B vs A $\Delta$ QWK | **-0.0360** (95% CI [-0.0537, -0.0183]) | **-0.0262** (95% CI [-0.0456, -0.0067]) |
| Paired $t(29)$ | -4.159 ($p = 0.000259$) | -2.749 ($p = 0.01018$) |
| Wilcoxon $W = \min(W^+, W^-)$ | **71.5** ($W^+=71.5, W^-=393.5$) | **125.0** ($W^+=125.0, W^-=340.0$) |
| Exact Cond. Permutation $p$ | **0.000514** | **0.025888** |
| Holm-Adjusted $p_{\text{adj}}$ | **0.000514** | **0.025888** |
