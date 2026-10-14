"""
generate_phase8_claim_ledger.py
Phase 5.6 - Phase 8: Authoritative Scientific Claim Ledger (Peer-Review Hardened)
Produces:
- outputs/phase56_publication_lock/claim_ledger_final.csv
- outputs/phase56_publication_lock/submission_package/tables/claim_ledger_final.csv
"""

import os
import csv
from datetime import datetime

OUT_CSV = "outputs/phase56_publication_lock/claim_ledger_final.csv"
SUB_CSV = "outputs/phase56_publication_lock/submission_package/tables/claim_ledger_final.csv"

claims = [
    {
        "claim_id": "CLM-001",
        "topic": "Monotonic CORAL Architecture",
        "manuscript_statement": "The MonotonicCORALHead enforces monotonically decreasing cutpoint biases b_0 > b_1 > ... > b_{K-2} via softplus parameterization, guaranteeing non-negative class probabilities summing to 1.0 and rank consistency.",
        "supporting_artifacts": "outputs/casm_v2/casm_v2_corrected_modules.py; tests/test_phase56_regression_hardening.py",
        "verification_command": "pytest tests/test_phase56_regression_hardening.py -k test_monotonic_coral_head_properties",
        "recomputed_result": "Biases strictly decreasing: b_0 > b_1 > b_2 > b_3; Class probabilities in [0, 1] and sum to 1.0 within 1e-6 tolerance across all batches.",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Mathematical guarantee applies strictly to logits and probabilities; actual sorting accuracy depends on learned feature representations."
    },
    {
        "claim_id": "CLM-002",
        "topic": "Historical Image-Only Baseline Performance",
        "manuscript_statement": "The image-only baseline (Config A) achieves a mean Quadratic Weighted Kappa (QWK) of 0.8478 ± 0.0260 and mean MAE of 0.1145 ± 0.0178 across 30 prespecified random seeds on Historical Split V2 (N_test = 12,833).",
        "supporting_artifacts": "outputs/casm_v2/phase50_per_seed_predictions/preds_A_seed*.pt; outputs/phase56_publication_lock/statistics_recomputed.json",
        "verification_command": "python outputs/phase56_publication_lock/generate_phase2_statistical_audit.py",
        "recomputed_result": "Mean QWK = 0.8478 ± 0.0260; Mean MAE = 0.1145 ± 0.0178; Mean Accuracy = 92.11% ± 1.49% (N_test=12,833 images, n_seeds=30).",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Must be transparently disclosed as evaluating on Historical Split V2, which contains 1 cross-split session (20200730)."
    },
    {
        "claim_id": "CLM-003",
        "topic": "Negative Effect of Species Conditioning",
        "manuscript_statement": "Auxiliary fruit species conditioning (Config B) significantly degrades ordinal grading compared to visual features alone across evaluated seeds (Mean diff = -0.0360 QWK, paired t(29) = -4.159, p = 0.000259, Wilcoxon W = min(W+, W-) = 71.5 with W+ = 71.5, exact conditional permutation p = 0.000514, Holm-adjusted p_adj = 0.000514).",
        "supporting_artifacts": "outputs/casm_v2/phase50_per_seed_predictions/preds_[A,B]_seed*.pt; outputs/phase56_publication_lock/statistics_recomputed.csv",
        "verification_command": "python outputs/phase56_publication_lock/generate_phase2_statistical_audit.py",
        "recomputed_result": "Mean diff = -0.035993; 95% CI [-0.0537, -0.0183]; t(29) = -4.1590; p_t_raw = 2.59e-4; Holm p_t_adj = 2.59e-4; Wilcoxon W = 71.5 (W+=71.5, W-=393.5); Exact perm p_w = 5.14e-4; Holm p_w_adj = 5.14e-4; SciPy exact = 5.55e-4; Asymptotic = 9.28e-4; Pairs 9+/21-.",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Wilcoxon statistic is explicitly defined as min(W+, W-) = 71.5 (fractional rank 13.5 due to tie between Seed 3 and Seed 73). Exact conditional permutation test under observed ranks confirms p = 0.000514."
    },
    {
        "claim_id": "CLM-004",
        "topic": "Negative Effect of OOD and Context Fusion",
        "manuscript_statement": "Concatenating open-set detection signals (Config C) or joint species-OOD context (Config D) causes even severe degradation (C-A: diff = -0.0700 QWK, t(29) = -7.091, p = 8.41e-8, W = 16.0, exact p = 3.15e-7; D-A: diff = -0.0848 QWK, t(29) = -10.797, p = 1.13e-11, W = 0.0, exact p = 1.86e-9).",
        "supporting_artifacts": "outputs/casm_v2/phase50_per_seed_predictions/preds_[A,C,D]_seed*.pt; outputs/phase56_publication_lock/statistics_recomputed.csv",
        "verification_command": "python outputs/phase56_publication_lock/generate_phase2_statistical_audit.py",
        "recomputed_result": "C-A: diff = -0.069997, 95% CI [-0.0902, -0.0498], t(29) = -7.091, Holm p_t = 1.68e-7, W = 16.0, exact p = 3.15e-7; D-A: diff = -0.084780, 95% CI [-0.1008, -0.0687], t(29) = -10.797, Holm p_t = 3.38e-11, W = 0.0, exact p = 1.86e-9 (0 pos / 30 neg).",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Acknowledge as a critical negative scientific finding: auxiliary tabular metadata introduces feature interference in end-to-end ordinal heads."
    },
    {
        "claim_id": "CLM-005",
        "topic": "Statistical Multiplicity Correction",
        "manuscript_statement": "Under prespecified Holm-Bonferroni family-wise error rate control across the m=3 comparison family {B-A, C-A, D-A}, all performance degradations remain statistically significant at alpha = 0.05.",
        "supporting_artifacts": "outputs/phase56_publication_lock/statistics_recomputed.json; outputs/phase56_publication_lock/statistical_audit.md",
        "verification_command": "pytest tests/test_phase56_regression_hardening.py -k test_holm_bonferroni_multiplicity_control",
        "recomputed_result": "Historical Holm-adjusted p-values: t-test: B-A=2.59e-4, C-A=1.68e-7, D-A=3.38e-11; Wilcoxon exact perm: B-A=5.14e-4, C-A=6.29e-7, D-A=5.59e-9. All p_adj << 0.05 across n_seeds=30.",
        "status": "VERIFIED",
        "limitations_and_required_wording": "The comparison family is prespecified strictly as the 3 candidate enhancements against the Image-Only baseline within each protocol."
    },
    {
        "claim_id": "CLM-006",
        "topic": "Replication on Strict Session-Held-Out Split",
        "manuscript_statement": "In a 100% group session-held-out protocol (N_test = 12,980 test images, 0 cross-split sessions across all 80 sessions), the visual-only baseline achieves 0.8175 ± 0.0354 QWK across 30 seeds, and auxiliary conditioning remains significantly worse (B-A: diff = -0.0262, t(29) = -2.749, p = 0.01018, W = 125.0, exact perm p = 0.025888, Holm p_adj = 0.025888).",
        "supporting_artifacts": "outputs/phase52_final_validation/session_holdout_retrain/per_seed_results.csv; outputs/phase56_publication_lock/statistics_recomputed.csv",
        "verification_command": "python outputs/phase56_publication_lock/generate_phase2_statistical_audit.py",
        "recomputed_result": "Config A QWK = 0.8175 ± 0.0354; B-A: diff = -0.026153, 95% CI [-0.0456, -0.0067], t(29) = -2.7490, p_t_raw = 0.01018, Holm p_t_adj = 0.01018; Wilcoxon W = 125.0 (W+=125.0, W-=340.0), exact perm p_w = 0.025888, Holm p_w_adj = 0.025888, SciPy exact = 0.026229.",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Demonstrates persistence of the negative effect under strict session isolation, ruling out session clustering artifacts on the evaluated dataset."
    },
    {
        "claim_id": "CLM-007",
        "topic": "Historical Session Leakage Characterization",
        "manuscript_statement": "In the historical split V2, acquisition session 20200730 was split between train (147 images) and test (119 images). Reallocating all 266 images of this session to the test set yielded the 12,980-sample zero-leakage test set.",
        "supporting_artifacts": "outputs/casm_v2/phase50_split_v2_manifest.csv; outputs/phase52_final_validation/session_holdout_manifest.csv; outputs/phase56_publication_lock/split_integrity_summary.csv",
        "verification_command": "python outputs/phase56_publication_lock/verify_split_integrity.py",
        "recomputed_result": "Verified: session 20200730 split 147 train / 119 test in split_v2. Sanitized in session_holdout_manifest: train decreases 39,790 -> 39,643 (-147), test increases 12,833 -> 12,980 (+147). Total: 56,036.",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Maintains clear distinction between path independence (0 leaks in both), bitwise hash independence, and session/turntable independence."
    },
    {
        "claim_id": "CLM-008",
        "topic": "Real-Image Open-Set Recognition Performance",
        "manuscript_statement": "On 5,000 real images of 5 unencountered fruit species (Cherry, Kiwi, Peach, Pineapple, Plum), Mahalanobis distance achieves 84.22% AUROC (53.28% AUPR, 38.39% FPR@95%), Energy score achieves 81.99% AUROC, and MSP achieves 74.26% AUROC.",
        "supporting_artifacts": "outputs/casm_v2/phase50_ood_features/ood_real_features.pt; outputs/final_audit/ood_predictions.csv; outputs/casm_v2/phase50_ood_benchmark.csv",
        "verification_command": "python outputs/phase56_publication_lock/revalidate_ood.py",
        "recomputed_result": "Mahalanobis: AUROC 84.22%, AUPR 53.28%, FPR95 38.39%; Energy: AUROC 81.99%, AUPR 49.96%, FPR95 41.38%; MSP: AUROC 74.26%, AUPR 43.16%, FPR95 47.85%.",
        "status": "VERIFIED",
        "limitations_and_required_wording": "AUPR strictly treats OOD (is_ood=1) as the positive class. Synthetic 100% AUROC claims are permanently retracted."
    },
    {
        "claim_id": "CLM-009",
        "topic": "Domain-Dependent Detector Inversion",
        "manuscript_statement": "Mahalanobis distance collapses on real market bins (18.08% AUROC) and industrial conveyors (18.81% AUROC) while excelling on clean studio images (93.26%), whereas Free Energy remains resilient (92.49% on market, 98.89% on conveyor).",
        "supporting_artifacts": "outputs/casm_v2/phase50_ood_report.md; outputs/phase56_publication_lock/ood_audit.md",
        "verification_command": "python outputs/phase56_publication_lock/revalidate_ood.py",
        "recomputed_result": "Verified: Studio AUROC = 93.26% (Maha) vs 79.99% (Energy); Market AUROC = 18.08% (Maha) vs 92.49% (Energy); Conveyor AUROC = 18.81% (Maha) vs 98.89% (Energy).",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Characterized strictly as an observed distribution-dependent performance inversion; background causality is bounded as a hypothesis without controlled foreground segmentation ablations."
    },
    {
        "claim_id": "CLM-010",
        "topic": "Calibration and Selective Abstention",
        "manuscript_statement": "For exemplar Seed 83, selective abstention at 90% coverage reduces ordinal MAE from 0.0889 to 0.0136 (an exact 84.70% error reduction). Across 30 seeds, mean selective MAE is 0.0217 ± 0.0053 (81.1% mean reduction). Multi-seed mean ECE is 2.11% ± 0.53%, and ensemble ECE is 2.51%.",
        "supporting_artifacts": "outputs/casm_v2/phase50_per_seed_predictions/preds_A_seed83.pt; outputs/phase56_publication_lock/calibration_recomputed.csv; outputs/phase56_publication_lock/selective_prediction_recomputed.csv",
        "verification_command": "python outputs/phase56_publication_lock/generate_phase3_calibration_audit.py",
        "recomputed_result": "Seed 83 baseline cutpoint MAE=0.0889 -> 0.0136 at 90% cov: (0.0889-0.0136)/0.0889 = 84.70%; 30-seed mean baseline cutpoint MAE=0.1145 -> sel MAE=0.0217 (81.1% red); Seed 83 ECE=1.47%; 30-seed mean ECE=2.11%; Ensemble ECE=2.51%.",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Disclose that 0.0889 is the ordinal cutpoint baseline (0.1234 is categorical argmax); clarify that low pooled ECE is heavily influenced by the 85.5% studio proportion."
    },
    {
        "claim_id": "CLM-011",
        "topic": "Model Parameters and Inference Latency",
        "manuscript_statement": "The trainable Monotonic CORAL head comprises 626,180 parameters. On an NVIDIA GeForce RTX 3050 Laptop GPU, full end-to-end forward inference achieves 49.0 FPS (BS=1) and 90.4 FPS (BS=32) in FP16, and 27.0 FPS (BS=1) and 40.1 FPS (BS=32) in FP32.",
        "supporting_artifacts": "outputs/casm_v2/phase50_checkpoints/model_A_seed83_best.pth; outputs/phase54_discrepancy_resolution/issue5_benchmarks.json; outputs/phase56_publication_lock/architecture_and_benchmark_audit.md",
        "verification_command": "pytest tests/test_phase56_regression_hardening.py -k test_real_checkpoint_parameter_count",
        "recomputed_result": "Exact parameter count = 626,180. Latency verified: FP16 BS=1 is 20.43 ms (49.0 FPS), BS=32 is 11.07 ms/sample (90.4 FPS); FP32 BS=1 is 37.10 ms (27.0 FPS), BS=32 is 24.92 ms/sample (40.1 FPS).",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Throughput claims measure PyTorch forward inference under CUDA synchronization; industrial deployment requires camera acquisition, lighting stabilization, and robotic ejection integration."
    },
    {
        "claim_id": "CLM-012",
        "topic": "Comprehensive Regression Harness & Reproducibility",
        "manuscript_statement": "A regression harness of 28 automated unit tests confirms 100% concordance with all reported parameters, split isolations, probability properties, and recomputed statistical tests.",
        "supporting_artifacts": "tests/test_*.py; outputs/casm_v2/test_corrected_modules.py; outputs/phase56_publication_lock/test_results.txt",
        "verification_command": "pytest -v",
        "recomputed_result": "28/28 tests passed in 115.36 seconds with 0 failures, 0 errors, and 0 skips.",
        "status": "VERIFIED",
        "limitations_and_required_wording": "Distinguishes artifact reproducibility (fully certified from stored checkpoints and predictions) from training reproducibility (subject to GPU non-deterministic CUDA atomics)."
    },
    {
        "claim_id": "CLM-013",
        "topic": "Industrial Deployment & Causality Bounds",
        "manuscript_statement": "Claims that the system is commercially ready for conveyor deployment or that backgrounds causally drive detector failure are bounded as empirical hypotheses.",
        "supporting_artifacts": "paper/manuscript.md; outputs/phase56_publication_lock/statistical_audit.md; outputs/phase56_publication_lock/ood_audit.md",
        "verification_command": "Forensic editorial inspection of paper/manuscript.md",
        "recomputed_result": "Commercial readiness and causal attribution claims removed or explicitly framed as hypotheses and engineering limitations.",
        "status": "PARTIALLY VERIFIED",
        "limitations_and_required_wording": "Framed as an academic proof-of-concept; physical deployment requires further study of lighting, camera synchronization, and mechanical sorting."
    }
]

def main():
    print("Generating Phase 8: Authoritative Scientific Claim Ledger...")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(SUB_CSV), exist_ok=True)

    fieldnames = [
        "claim_id", "topic", "manuscript_statement", "supporting_artifacts",
        "verification_command", "recomputed_result", "status", "limitations_and_required_wording"
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(claims)
    print(f"Saved {OUT_CSV}")

    with open(SUB_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(claims)
    print(f"Saved {SUB_CSV}")

if __name__ == "__main__":
    main()
