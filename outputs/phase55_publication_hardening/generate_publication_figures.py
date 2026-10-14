"""
generate_publication_figures.py - Phase 5.5 Section 8: Publication-Quality Figures Generator
Generates publication-ready figures (300 DPI, IEEE-styled) directly from audited numerical artifacts:
1. figure1_dataset_composition.png: Sample distribution by acquisition domain & severity grade
2. figure2_qwk_distribution.png: 30-seed boxplots for Configs A, B, C, D (Historical & Session-Held-Out)
3. figure3_paired_contrasts.png: Paired difference distributions (B-A, C-A, D-A) confirming degradation
4. figure4_ood_domain_inversion.png: Mahalanobis collapse vs Energy Score resilience across domains
5. figure5_selective_risk_coverage.png: Monotonic risk-coverage curves for selective prediction
6. figure6_inference_benchmarks.png: Head-only vs Full Pipeline throughput & latency on RTX 3050 GPU
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = "outputs/phase55_publication_hardening/figures"
STAT_JSON = "outputs/phase55_publication_hardening/statistical_final.json"
OOD_JSON = "outputs/phase54_discrepancy_resolution/issue4_ood_metrics_recomputation.json"
SEL_JSON = "outputs/phase55_publication_hardening/selective_metric_reconciliation.json"
BENCH_JSON = "outputs/phase54_discrepancy_resolution/issue5_benchmarks.json"
MANIFEST_PATH = "outputs/casm_v2/phase50_split_v2_manifest.csv"

# Configure IEEE style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def make_fig1():
    df = pd.read_csv(MANIFEST_PATH)
    ct = pd.crosstab(df["domain"], df["severity_grade"])
    domains = ct.index.tolist()
    grades = ["Grade 0 (Healthy)", "Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)", "Grade 4 (Rotten)"]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]
    
    bottom = np.zeros(len(domains))
    for g in range(5):
        vals = ct[g].values if g in ct.columns else np.zeros(len(domains))
        ax.bar(domains, vals, bottom=bottom, label=grades[g], color=colors[g], edgecolor="black", linewidth=0.5, alpha=0.9)
        bottom += vals
        
    ax.set_ylabel("Number of Samples")
    ax.set_title("Dataset Composition by Acquisition Domain & Severity Grade")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    
    out_path = os.path.join(FIG_DIR, "figure1_dataset_composition.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def make_fig2_and_3():
    with open(STAT_JSON, "r") as f:
        stat_data = json.load(f)
        
    # Fig 2: QWK by Config across protocols
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    
    configs = ["A", "B", "C", "D"]
    labels = ["A (Image)", "B (+Species)", "C (+OOD)", "D (All)"]
    
    # Historical
    hist_means = [stat_data["historical_split_v2"]["summary_by_config"][c]["qwk_mean"] for c in configs]
    hist_sds = [stat_data["historical_split_v2"]["summary_by_config"][c]["qwk_sd"] for c in configs]
    
    ax1.bar(labels, hist_means, yerr=hist_sds, capsize=4, color="#34495e", edgecolor="black", alpha=0.85)
    ax1.set_title("Historical Split V2 (N=12,833)")
    ax1.set_ylabel("Quadratic Weighted Kappa (QWK)")
    ax1.set_ylim(0.65, 0.90)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(hist_means):
        ax1.text(i, v - 0.04, f"{v:.3f}", ha="center", color="white", fontweight="bold", fontsize=9)
        
    # Session Holdout
    retrain_means = [stat_data["session_held_out_retrain"]["summary_by_config"][c]["qwk_mean"] for c in configs]
    retrain_sds = [stat_data["session_held_out_retrain"]["summary_by_config"][c]["qwk_sd"] for c in configs]
    
    ax2.bar(labels, retrain_means, yerr=retrain_sds, capsize=4, color="#2980b9", edgecolor="black", alpha=0.85)
    ax2.set_title("100% Group Session Holdout (N=12,980)")
    ax2.set_ylim(0.65, 0.90)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(retrain_means):
        ax2.text(i, v - 0.04, f"{v:.3f}", ha="center", color="white", fontweight="bold", fontsize=9)
        
    fig.suptitle("Grading Performance Across Configurations (30 Seeds Mean ± SD)", y=1.02)
    out_path = os.path.join(FIG_DIR, "figure2_qwk_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")
    
    # Fig 3: Paired Differences
    fig, ax = plt.subplots(figsize=(6, 4))
    contrasts = ["B vs A", "C vs A", "D vs A"]
    
    hist_diffs = [next(r["mean_diff"] for r in stat_data["historical_split_v2"]["contrasts"] if r["contrast"]==c) for c in contrasts]
    hist_cis = [next(r["ci_95"] for r in stat_data["historical_split_v2"]["contrasts"] if r["contrast"]==c) for c in contrasts]
    hist_errs = [[m - ci[0], ci[1] - m] for m, ci in zip(hist_diffs, hist_cis)]
    hist_errs = np.array(hist_errs).T
    
    retrain_diffs = [next(r["mean_diff"] for r in stat_data["session_held_out_retrain"]["contrasts"] if r["contrast"]==c) for c in contrasts]
    retrain_cis = [next(r["ci_95"] for r in stat_data["session_held_out_retrain"]["contrasts"] if r["contrast"]==c) for c in contrasts]
    retrain_errs = [[m - ci[0], ci[1] - m] for m, ci in zip(retrain_diffs, retrain_cis)]
    retrain_errs = np.array(retrain_errs).T
    
    x = np.arange(len(contrasts))
    width = 0.35
    
    ax.bar(x - width/2, hist_diffs, width, yerr=hist_errs, capsize=4, label="Historical Split V2", color="#7f8c8d", edgecolor="black")
    ax.bar(x + width/2, retrain_diffs, width, yerr=retrain_errs, capsize=4, label="Session Holdout", color="#c0392b", edgecolor="black")
    
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(contrasts)
    ax.set_ylabel("Paired Difference in QWK (ΔQWK)")
    ax.set_title("Paired Contrasts vs Image-Only Baseline (95% CI)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    
    out_path = os.path.join(FIG_DIR, "figure3_paired_contrasts.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def make_fig4():
    with open(OOD_JSON, "r") as f:
        ood_data = json.load(f)
        
    domains = ["fruits360", "real_quality", "real_grade", "real_severity"]
    labels = ["fruits360\n(Studio)", "real_quality\n(Market Bins)", "real_grade\n(Conveyor)", "real_severity\n(Field)"]
    
    mahal = [ood_data["domain_stratified_metrics"][d]["Mahalanobis"]["auroc"] for d in domains]
    energy = [ood_data["domain_stratified_metrics"][d]["Energy"]["auroc"] for d in domains]
    msp = [ood_data["domain_stratified_metrics"][d]["MSP"]["auroc"] for d in domains]
    
    x = np.arange(len(domains))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, mahal, width, label="Mahalanobis Distance", color="#e74c3c", edgecolor="black")
    ax.bar(x, energy, width, label="Energy Score (Logit-space)", color="#27ae60", edgecolor="black")
    ax.bar(x + width, msp, width, label="Max Softmax Prob (MSP)", color="#2980b9", edgecolor="black")
    
    ax.axhline(50, color="gray", linestyle=":", label="Random Guessing (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("OOD Detection AUROC (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Open-Set Detector Performance Across Acquisition Domains")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    
    out_path = os.path.join(FIG_DIR, "figure4_ood_domain_inversion.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def make_fig5():
    calib_json = "outputs/phase54_discrepancy_resolution/issue3_calibration_traceability.json"
    with open(calib_json, "r") as f:
        cal_data = json.load(f)
        
    curve = cal_data["single_run_exemplar"]["selective_abstention_curve"]
    covs = [c["coverage_target"] * 100.0 for c in curve]
    accs = [c["selective_accuracy"] * 100.0 for c in curve]
    maes = [c["selective_mae"] for c in curve]
    
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    
    color = "#2980b9"
    ax1.set_xlabel("Retained Coverage (%)")
    ax1.set_ylabel("Selective Accuracy (%)", color=color)
    line1 = ax1.plot(covs, accs, marker="o", color=color, linewidth=2, label="Selective Accuracy")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(90, 101)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color = "#c0392b"
    ax2.set_ylabel("Selective MAE", color=color)
    line2 = ax2.plot(covs, maes, marker="s", linestyle="--", color=color, linewidth=2, label="Selective MAE")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(-0.01, 0.15)
    
    # Invert x axis so that decreasing coverage is reading left-to-right (abstaining more)
    ax1.set_xlim(102, 48)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", frameon=True)
    ax1.set_title("Selective Prediction Risk-Coverage Trade-off (Seed 83 Exemplar)")
    
    out_path = os.path.join(FIG_DIR, "figure5_selective_risk_coverage.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def make_fig6():
    with open(BENCH_JSON, "r") as f:
        bench_data = json.load(f)
        
    benches = bench_data["benchmarks"]
    e2e_fp16 = [b for b in benches if b["pipeline_scope"].startswith("Full") and b["precision"]=="FP16"]
    e2e_fp32 = [b for b in benches if b["pipeline_scope"].startswith("Full") and b["precision"]=="FP32"]
    
    bs = [1, 16, 32]
    fps_fp16 = [next(b["throughput_fps"] for b in e2e_fp16 if b["batch_size"]==s) for s in bs]
    fps_fp32 = [next(b["throughput_fps"] for b in e2e_fp32 if b["batch_size"]==s) for s in bs]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(bs))
    width = 0.35
    
    ax.bar(x - width/2, fps_fp16, width, label="Full Pipeline (FP16)", color="#27ae60", edgecolor="black")
    ax.bar(x + width/2, fps_fp32, width, label="Full Pipeline (FP32)", color="#7f8c8d", edgecolor="black")
    
    # Conveyor standard band
    ax.axhspan(10, 30, color="#f39c12", alpha=0.25, label="Optical Sorter Conveyor Speed (10-30 fruits/s)")
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"BS={s}" for s in bs])
    ax.set_ylabel("Inference Throughput (FPS)")
    ax.set_title("End-to-End Pipeline Inference Throughput (RTX 3050 GPU)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    
    for i, v in enumerate(fps_fp16):
        ax.text(i - width/2, v + 2, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(fps_fp32):
        ax.text(i + width/2, v + 2, f"{v:.1f}", ha="center", fontsize=9)
        
    ax.set_ylim(0, 110)
    out_path = os.path.join(FIG_DIR, "figure6_inference_benchmarks.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def main():
    print("Executing Phase 5.5 Section 8: Publication-Quality Figures Generation...")
    os.makedirs(FIG_DIR, exist_ok=True)
    make_fig1()
    make_fig2_and_3()
    make_fig4()
    make_fig5()
    make_fig6()
    print("All 6 publication figures generated successfully in outputs/phase55_publication_hardening/figures/")

if __name__ == "__main__":
    main()
