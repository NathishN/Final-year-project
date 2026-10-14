# Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading
## Phase 5.6 Final Forensic Publication Package

**Title:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation  
**Target Venue:** IEEE Transactions on Agri-Food Computing / Pattern Analysis & Machine Intelligence  
**Artifact Directory:** `outputs/phase56_publication_lock/submission_package/`  
**Certification Status:** **READY FOR SUBMISSION** (All numerical claims and statistical tests forensically verified)  
**Date:** October 2026  

---

### Package Contents

```
submission_package/
├── manuscript.md                       <- Authoritative manuscript with audited numbers
├── README.md                           <- Reproduction guide and package index
├── FINAL_PUBLICATION_AUDIT.md          <- Final sign-off verdict and consistency certification
├── environment_info.txt                <- Hardware, CUDA, and library version manifest
├── test_results.txt                    <- Complete pytest execution log (28/28 tests passed)
├── figures/                            <- 300 DPI publication figures (Figure 1 through 6)
│   ├── figure1_dataset_composition.png
│   ├── figure2_qwk_distribution.png
│   ├── figure3_paired_contrasts.png
│   ├── figure4_ood_domain_inversion.png
│   ├── figure5_selective_risk_coverage.png
│   └── figure6_inference_benchmarks.png
├── tables/                             <- Authoritative CSV tables matching manuscript text
│   ├── authoritative_statistical_table.csv
│   ├── claim_ledger_final.csv
│   ├── figure_manifest.csv
│   ├── split_integrity_summary.csv
│   └── ood_metrics_recomputed.csv
└── audits/                             <- Forensic audit reports for each sub-topic
    ├── artifact_inventory.md
    ├── statistical_audit.md
    ├── calibration_and_abstention_audit.md
    ├── split_integrity_audit.md
    ├── ood_audit.md
    ├── architecture_and_benchmark_audit.md
    └── reproducibility_audit.md
```

---

### Exact Reproduction Instructions

To reproduce all results, statistics, and figures from the repository root:

```bash
# 1. Execute automated regression harness (28 tests)
pytest -v

# 2. Recompute paired statistical tests & resolve Wilcoxon discrepancy
python outputs/phase56_publication_lock/generate_phase2_statistical_audit.py

# 3. Recompute calibration & selective abstention curves
python outputs/phase56_publication_lock/generate_phase3_calibration_audit.py

# 4. Audit dataset split manifests & session isolation
python outputs/phase56_publication_lock/verify_split_integrity.py

# 5. Revalidate real-image OOD detection metrics
python outputs/phase56_publication_lock/revalidate_ood.py

# 6. Regenerate all 300 DPI publication figures
python outputs/phase56_publication_lock/generate_all_publication_figures.py
```

---

### Key Scientific Findings Summary

1. **Monotonic CORAL Parameterization:** Structurally guarantees decreasing cutpoint biases $b_0 > b_1 > \dots > b_{K-2}$, preventing threshold inversions and producing class probabilities that strictly sum to 1.0 unconditionally.
2. **Robust Negative Result:** Across $n=30$ seeds in both Historical ($N_{\text{test}}=12,833$) and 100% Group Session-Held-Out ($N_{\text{test}}=12,980$) protocols ($N=240$ total runs), auxiliary species conditioning, OOD conditioning, and joint context conditioning all significantly degrade ordinal QWK ($p_{\text{adj}} < 0.05$ across all paired tests under Holm–Bonferroni multiplicity control).
3. **Forensic Wilcoxon Resolution:** The historical B vs A statistic is explicitly defined as $W = \min(W^+, W^-) = 71.5$ ($W^+ = 71.5, W^- = 393.5$, due to fractional tie rank $13.5$ at $|\Delta|=0.0263$). Under the true conditional permutation test conditioned on observed ranks, $p = 0.000514$ ($276,128 / 1,073,741,824$ one-sided). SciPy's integer no-ties lookup yields $p = 0.000555$, and asymptotic normal approximation yields $p = 0.000928$.
4. **Selective Abstention:** Discarding the lowest 10% confidence predictions on representative Seed 83 raises accuracy from 93.71% to 99.28% and cuts ordinal cutpoint MAE from 0.0889 to 0.0136 (an exact 84.70% error reduction).
5. **Real-Image OOD Evaluation:** On 5,000 real images across 5 unseen species, Mahalanobis distance collapses to 18.08% on market-bin images due to background shift, whereas Free Energy Score maintains $>92\%$ AUROC across all natural domains.
