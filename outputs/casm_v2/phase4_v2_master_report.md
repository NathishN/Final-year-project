# Phase 4.2 — CASM v2 Implementation and Scientific Validation Master Report

## Executive Summary
Phase 4.2 successfully designed, validated, and benchmarked **CASM v2 (Selective Species & OOD Fusion Network)** across 5 random seeds (42, 123, 777, 999, 2025).

## Master Benchmark Summary Table

| Model ID | Configuration | Acc Mean ± Std | MAE Mean ± Std | QWK Mean ± Std | Macro-F1 ± Std | OBO Acc ± Std |
|---|---|---:|---:|---:|---:|---:|
| **A** | Image Only (Baseline) | 87.98% ± 0.69 | 0.2097 ± 0.0311 | **0.7249 ± 0.0521** | 29.74% ± 2.11 | 94.54% ± 1.02 |
| **B** | Image + Species | 86.21% ± 1.95 | 0.2574 ± 0.0557 | **0.7023 ± 0.0604** | 26.01% ± 2.32 | 93.29% ± 1.79 |
| **C** | Image + OOD | 87.93% ± 1.03 | 0.2112 ± 0.0301 | **0.7113 ± 0.0346** | 30.99% ± 1.36 | 94.60% ± 0.72 |
| **D** | CASM v2 (Img + Spec + OOD) | 87.37% ± 2.13 | 0.2094 ± 0.0326 | **0.7423 ± 0.0374** | 29.64% ± 3.03 | 94.51% ± 1.80 |
| **E** | Full CASM (v1 Legacy) | 87.71% ± 0.79 | 0.2096 ± 0.0160 | **0.7448 ± 0.0161** | 29.07% ± 2.29 | 94.75% ± 0.76 |

## Generated Phase 4.2 Deliverables Checklist
1. [`outputs/casm_v2/dataset_audit.md`](file:///c:/fruit_dataset/outputs/casm_v2/dataset_audit.md)
2. [`outputs/casm_v2/dataset_statistics.json`](file:///c:/fruit_dataset/outputs/casm_v2/dataset_statistics.json)
3. [`outputs/casm_v2/validation_report.md`](file:///c:/fruit_dataset/outputs/casm_v2/validation_report.md)
4. [`outputs/casm_v2/casm_v2_design.md`](file:///c:/fruit_dataset/outputs/casm_v2/casm_v2_design.md)
5. [`outputs/casm_v2/context_ablation_results.csv`](file:///c:/fruit_dataset/outputs/casm_v2/context_ablation_results.csv)
6. [`outputs/casm_v2/context_significance_report.md`](file:///c:/fruit_dataset/outputs/casm_v2/context_significance_report.md)
7. [`outputs/casm_v2/context_importance_report.md`](file:///c:/fruit_dataset/outputs/casm_v2/context_importance_report.md)
8. [`outputs/casm_v2/architecture_diagnostics.md`](file:///c:/fruit_dataset/outputs/casm_v2/architecture_diagnostics.md)
9. [`outputs/casm_v2/domain_robustness_report.md`](file:///c:/fruit_dataset/outputs/casm_v2/domain_robustness_report.md)
10. [`outputs/casm_v2/final_research_decision.md`](file:///c:/fruit_dataset/outputs/casm_v2/final_research_decision.md)
11. [`outputs/casm_v2/ieee_review_report.md`](file:///c:/fruit_dataset/outputs/casm_v2/ieee_review_report.md)
12. [`outputs/casm_v2/phase4_v2_master_report.md`](file:///c:/fruit_dataset/outputs/casm_v2/phase4_v2_master_report.md)
