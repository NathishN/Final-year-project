# Phase 5.6 Final Submission Certification & Forensic Audit Report

**Authority:** Senior ML Research Engineer, Statistical Auditor & Scientific Manuscript Editor  
**Project:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation  
**Directory:** `./` (Repository Root)  
**Artifact Archive:** `outputs/final_submission/`  
**Evaluation Platform:** Windows 11 Enterprise, Python 3.12.4, PyTorch 2.5.1+cu121, NVIDIA RTX 3050 Laptop GPU (6GB)  
**Audit Timestamp:** 2026-10-14 14:43:00  

---

## 1. Final Submission Verdict

# **READY FOR SUBMISSION**

All material experimental results, paired statistical hypothesis tests, split manifests, OOD detection metrics, calibration definitions, model parameters, and hardware latency benchmarks are 100% verified and traced to underlying immutable artifacts.

Zero contradictions, unverified claims, or mathematical inconsistencies remain in the manuscript or submission package.

*(Note for Authors: Author names and affiliations are currently formatted as anonymized placeholders for double-blind peer review. If submitting to a single-blind venue, insert author names and institutional affiliations directly into `manuscript.md` line 3–4.)*

---

## 2. Summary of Resolved Critical Discrepancies

| Item / Finding | Previous Concern / Inconsistency | Authoritative Forensic Resolution | Status |
|:---|:---|:---|:---:|
| **Critical Discrepancy A: Changing QWK Values** | Earlier reports cited A: 0.8478, B: 0.8118, C: 0.7778, D: 0.7630 for historical, and A: 0.8038, B: 0.7776, C: 0.7513, D: 0.7164 for session holdout. Recent drafts mistakenly transcribed single-seed numbers (Seed 37 A=0.8175) or preliminary tables. | Directly loaded and computed metrics across all 240 raw `.pt` prediction files. Re-established the true 30-seed tensor means as the single source of truth in `final_results.json`: Historical A=0.8478, B=0.8118, C=0.7778, D=0.7630; Session Holdout A=0.8038, B=0.7776, C=0.7513, D=0.7164. All paired differences (-0.0360, -0.0700, -0.0848, and -0.0262, -0.0525, -0.0874) match the individual tensor evaluations to 4 decimal places. | **VERIFIED & RESOLVED** |
| **Exact Wilcoxon Signed-Rank Test with Ties** | SciPy `method='exact'` evaluated $W=72$ against an integer no-ties distribution ($p=0.000555$). Reviewer correctly noted that ties invalidate standard no-ties distributions. | Seed 3 and Seed 73 tie at $|\Delta|=0.0263$ (average rank 13.5). Positive rank sum is $W^+=71.5$, negative rank sum is $W^-=393.5$, $W=\min(W^+,W^-)=71.5$. Implemented dynamic programming exact conditional permutation test enumerating all $2^{30} = 1,073,741,824$ sign assignments under observed ranks with ties: exactly 276,128 combinations produce $W^+ \le 71.5$, yielding **$p = 0.000514$**. Reported side-by-side with asymptotic approximation ($p = 0.000928$). | **VERIFIED & RESOLVED** |
| **Statistical Significance Statement** | Earlier report overclaimed that all results are significant at $\alpha = 0.001$, but session-held-out B-A has $p \approx 0.010 - 0.027$. | Corrected all statements: Historical contrasts are significant at $\alpha = 0.001$ ($p_{\text{adj}} \le 0.000514$). Session-held-out auxiliary species conditioning (B vs A) is significant at $\alpha = 0.05$ (paired $t(29) = -2.749, p_{\text{adj}} = 0.01018$; Wilcoxon exact perm $p_{\text{adj}} = 0.025888$). Never claim $p < 0.001$ when $p > 0.001$. | **VERIFIED & RESOLVED** |
| **Sample-Size Notation** | Conflation of test-set image counts ($N_{\text{test}} = 12,833$ or $12,980$) with replication units ($n_{\text{seeds}} = 30$, $\text{df} = 29$). | Separated sample-size notation throughout text and tables: $N_{\text{test}}$ represents evaluated test-set image instances; $n_{\text{seeds}} = 30$ represents independent paired training seeds yielding degrees of freedom $\text{df} = 29$ in paired $t(29)$. | **VERIFIED & RESOLVED** |
| **Session Isolation Bounds** | Moving 266 samples from Session `20200730` eliminated session leakage, but universal dataset independence was overstated. | Verified 0 session leakage across all 80 sessions in `session_holdout_manifest.csv`. Qualified discussion: this eliminates session clustering on the studied dataset, but does not claim independence from unobserved domain or sensor variations. | **VERIFIED & RESOLVED** |
| **OOD Domain Breakdown & Inversion** | Pooled AUROC (84.22%) masked catastrophic failure on market bins (18.08%). Causal background attribution was unproven. | Displayed complete domain breakdown in Table 4 and Figure 4. Highlighted Free Energy Score resilience ($92.49\%$ on market bins, $98.89\%$ on conveyor). Bounded background failure as an observed empirical hypothesis, requiring foreground segmentation for causal attribution. | **VERIFIED & RESOLVED** |
| **Hardware Throughput Qualifications** | Stated throughput (49.0–90.4 FPS in FP16 on RTX 3050) implied commercial conveyor readiness. | Qualified benchmarks as isolated PyTorch forward inference under CUDA timing. Clarified that physical sorting lines require additional time for camera acquisition, illumination calibration, and robotic ejection. | **VERIFIED & RESOLVED** |
| **Test Suite Validity** | Passing tests did not prove underlying artifacts were genuine. | Verified that regression tests load real checkpoints (`model_A_seed83_best.pth`, confirming 626,180 parameters), real prediction tensors, and real 56,036-sample CSV manifests. All 28 tests pass in 107.86s. | **VERIFIED & RESOLVED** |
| **Submission Portability** | Unportable absolute local drive links existed in markdown documents. | Removed all local Windows drive URIs from `outputs/final_submission/` in favor of portable relative repository references (`./`). | **VERIFIED & RESOLVED** |

---

## 3. Authoritative Single Source of Truth Table

Evaluated over $n_{\text{seeds}} = 30$ independent training runs ($\text{df} = 29$):

| Split Protocol | Contrast | Baseline A QWK | Context QWK | $\Delta$ QWK (95% CI) | Paired $t(29)$ ($p_{\text{param}}$) | Wilcoxon $W = \min(W^+, W^-)$ | Exact Cond. $p_{\text{exact}}$ | Asymptotic $p_{\text{asymp}}$ | Holm $p_{\text{adj}}$ | Cohen's $d_z$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Historical `split_v2`**<br>($N_{\text{test}} = 12,833$) | **B vs A**<br>(Species) | $0.8478 \pm 0.0260$ | $0.8118 \pm 0.0364$ | $-0.0360$<br>([$-0.0537, -0.0183$]) | $-4.1590$<br>($0.000259$) | $71.5$<br>($W^+=71.5, W^-=393.5$) | **$0.000514$** | $0.000928$ | **$0.000514$** | $-0.7593$ |
| | **C vs A**<br>(OOD Dist) | $0.8478 \pm 0.0260$ | $0.7778 \pm 0.0450$ | $-0.0700$<br>([$-0.0902, -0.0498$]) | $-7.0911$<br>($8.41 \times 10^{-8}$) | $16.0$<br>($W^+=16.0, W^-=449.0$) | **$3.15 \times 10^{-7}$** | $8.47 \times 10^{-6}$ | **$6.30 \times 10^{-7}$** | $-1.2947$ |
| | **D vs A**<br>(Joint) | $0.8478 \pm 0.0260$ | $0.7630 \pm 0.0402$ | $-0.0848$<br>([$-0.1008, -0.0687$]) | $-10.7971$<br>($1.13 \times 10^{-11}$) | $0.0$<br>($W^+=0.0, W^-=465.0$) | **$1.86 \times 10^{-9}$** | $1.73 \times 10^{-6}$ | **$5.59 \times 10^{-9}$** | $-1.9713$ |
| **Group Session Holdout**<br>($N_{\text{test}} = 12,980$) | **B vs A**<br>(Species) | $0.8038 \pm 0.0335$ | $0.7776 \pm 0.0431$ | $-0.0262$<br>([$-0.0456, -0.0067$]) | $-2.7490$<br>($0.01018$) | $125.0$<br>($W^+=125.0, W^-=340.0$) | **$0.025888$** | $0.027025$ | **$0.025888$** | $-0.5019$ |
| | **C vs A**<br>(OOD Dist) | $0.8038 \pm 0.0335$ | $0.7513 \pm 0.0352$ | $-0.0525$<br>([$-0.0707, -0.0342$]) | $-5.8715$<br>($2.27 \times 10^{-6}$) | $29.0$<br>($W^+=29.0, W^-=436.0$) | **$3.24 \times 10^{-6}$** | $2.84 \times 10^{-5}$ | **$6.48 \times 10^{-6}$** | $-1.0720$ |
| | **D vs A**<br>(Joint) | $0.8038 \pm 0.0335$ | $0.7164 \pm 0.0494$ | $-0.0874$<br>([$-0.1081, -0.0667$]) | $-8.6231$<br>($1.70 \times 10^{-9}$) | $9.0$<br>($W^+=9.0, W^-=456.0$) | **$6.15 \times 10^{-8}$** | $4.29 \times 10^{-6}$ | **$1.84 \times 10^{-7}$** | $-1.5744$ |

*Multiplicity Control: Holm–Bonferroni across the family of three context configurations ($m=3$). All adjusted p-values remain $p_{\text{adj}} < 0.05$.*

---

## 4. Verification Check and Execution Log

All verification commands were executed from the repository root (`./`):

1. **Automated Regression Suite:**
   ```bash
   pytest -v
   ```
   *Result:* 28 passed in 107.86s, Exit code 0.

2. **Authoritative Results Compilation:**
   ```bash
   python scripts/generate_authoritative_final_results.py
   ```
   *Result:* Extracted 240 prediction tensors, generated `final_results.json` and `authoritative_statistical_table.csv`, Exit code 0.

3. **Publication Figures (300 DPI):**
   ```bash
   python scripts/generate_final_submission_figures.py
   ```
   *Result:* Generated `figure1` through `figure6` and `figure_manifest.csv`, Exit code 0.

4. **Manuscript Synchronization:**
   ```bash
   python scripts/compile_full_manuscript.py
   ```
   *Result:* Synchronized `paper/manuscript.md`, `outputs/final_submission/manuscript.md`, and `outputs/final_submission/abstract.md`, Exit code 0.

5. **Claim Ledger Audit:**
   ```bash
   python scripts/generate_authoritative_claim_ledger.py
   ```
   *Result:* Verified all 13 claims against `final_results.json`, Exit code 0.

---

## 5. Submission Package Contents (`outputs/final_submission/`)

- `manuscript.md`: Complete journal manuscript with all 16 required sections, verified references, and synchronized tables.
- `abstract.md`: Standalone abstract with exact audited numerical metrics.
- `final_results.json`: Single machine-readable source of truth containing all experimental results, contrasts, and hardware benchmarks.
- `tables/`:
  - `authoritative_statistical_table.csv`
  - `claim_ledger.csv`
  - `split_integrity_summary.csv`
  - `ood_metrics_recomputed.csv`
  - `calibration_recomputed.csv`
  - `selective_prediction_recomputed.csv`
  - `figure_manifest.csv`
- `figures/`: High-resolution 300 DPI figures (`figure1` through `figure6`).
- `references.bib`: Complete BibTeX file with verified DOIs and author lists.
- `claim_ledger.csv`: Audited claim matrix with strict limitations and required wording.
- `reproducibility_guide.md`: Step-by-step reproduction instructions.
- `environment_info.txt`: Hardware and software environment specifications.
- `test_results.txt`: Test suite execution logs.
- `data_and_code_availability.md`: Data and code access policies.
- `CHANGELOG.md`: Detailed chronological progression of all audit phases.
- `FINAL_SUBMISSION_REPORT.md`: This authoritative verdict document.

**The final research package is locked, artifact-backed, mathematically validated, and certified for submission.**
