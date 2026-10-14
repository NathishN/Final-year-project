# Phase 4.5 — Definitive Research Decision & Publication Audit Verdict

**Auditing Panel:** Senior Machine Learning Research Engineer, Computer Vision Scientist, Statistical ML Auditor, IEEE Transactions Reviewer  
**Date:** October 2026  
**Repository:** `C:\fruit_dataset`  
**Deliverable File:** `outputs/casm_v2/phase45_final_research_decision.md`

---

## 1. Formal Research Classification and Verdict

### Formal Classification: **OUTCOME D — RESEARCH VALIDITY UNRESOLVED**
*(With Promising Mechanistic Evidence for Species Conditioning, but Fundamental Dataset, Benchmark, and Implementation Vulnerabilities That Preclude Publication Until Repaired)*

```
+----------------------------------------------------------------------------------------------------+
|                                 DECISION CLASSIFICATION EVALUATION                                 |
+-------------------+---------------------------------------------------+----------------------------+
| Classification    | Formal Definition                                 | Audit Decision             |
+-------------------+---------------------------------------------------+----------------------------+
| **Outcome A**     | Validated Improvement: Data, implementation, and  | **REJECTED**               |
|                   | evaluation are valid; prespecified analysis       | (Failed confirmatory test; |
|                   | supports superiority claim.                       | $p=0.0593$; FWER fail)     |
+-------------------+---------------------------------------------------+----------------------------+
| **Outcome B**     | Promising but Inconclusive: Results reproducible, | **SECONDARY FINDING**      |
|                   | but confirmatory evidence is insufficient.        | (Describes algorithm only, |
|                   |                                                   | ignores protocol flaws)    |
+-------------------+---------------------------------------------------+----------------------------+
| **Outcome C**     | No Demonstrated Benefit: Corrected experiments    | **REJECTED**               |
|                   | demonstrate no meaningful advantage.              | (Species effect is real,   |
|                   |                                                   | $\Delta=+0.0444, p=0.0023$)|
+-------------------+---------------------------------------------------+----------------------------+
| **Outcome D**     | Research Validity Unresolved: Important dataset,  | **ACCEPTED AS PRIMARY**    |
|                   | provenance, implementation, or evaluation         | (Ghost class, synthetic    |
|                   | problems remain unresolved.                       | OOD, domain masking)       |
+-------------------+---------------------------------------------------+----------------------------+
```

---

## 2. Core Justifications for Outcome D Selection

While Phase 4.4 classified the status as Outcome B based purely on statistical $p$-values, our exhaustive Phase 4.5 audit uncovered deeper foundational flaws in the dataset, benchmarks, and model architectures that elevate the verdict to **Outcome D**:

### 1. Fatal Test Set Defect: Ghost Class (Grade 2 Has $0 / 12,714$ Samples)
The manuscript claims to evaluate a 5-grade ordinal quality grading system ($\{0, 1, 2, 3, 4\}$). In reality:
- Grade 2 ("Moderate Defect") has **exactly 0 ground-truth samples** in the historical test split.
- This was caused by an unstratified session shuffle in `generate_master_csv.py` that allocated all three sessions containing `mixed` quality to `train` ($n=327$) and `val` ($n=196$).
- The test set evaluates strictly a 4-class problem ($\{0, 1, 3, 4\}$).
- Claiming publication-grade validation of a 5-grade ordinal model on a test set that lacks one of the intermediate grades is scientifically indefensible.

### 2. OOD Benchmark Provenance Defect (Synthetic Data Leak)
In `run_phase2_ood.py` (lines 45–63), the reported Phase 2 OOD benchmark ($100.0\%$ AUROC for Mahalanobis OOD) was generated using **synthetic random Gaussian numbers** (`np.random.randn() + 5.0`) because feature files were missing on disk! Real images in `open-set-dataset` (`Cherry`, `Kiwi`, `Peach`, `Pineapple`, `Plum`) have never been processed through the visual backbone. Claiming "trustworthy open-set recognition" based on synthetic Gaussian vectors is a severe publication hazard.

### 3. Severe Domain Confounding & Laboratory Masking
- The test set is **86.31% synthetic turntable images** (`fruits360`), where all models achieve $\approx 97.2\%$ accuracy because of uniform white backgrounds and 100% Grade 0 ground truth.
- On genuine market inspection images (`real_quality`), accuracy drops to **8.7% - 10.4%**, and MAE rises to **1.51 grades**.
- The reported aggregate accuracy ($86.3\% - 87.6\%$) gives a completely false impression of real-world operational competence.

### 4. Implementation Deficiencies in Baseline & Model
- The original `CORALHead` lacked cutpoint monotonicity constraints, allowing negative discrete class probabilities.
- `AuditModel` had no unknown-species fallback, crashing with `IndexError` on out-of-range or negative IDs.
- Phase 4.4 discarded all trained checkpoints and sample-level predictions from disk.

---

## 3. Algorithmic and Mechanistic Insights (What WAS Validated)

Despite the dataset and benchmark invalidity, the mathematical and algorithmic investigations yielded high-value insights:

1. **Species Conditioning is the Sole Effective Signal:**
   - In the 4-way ablation, Config B (Image + Species) achieved the highest overall QWK: **$0.7368 \pm 0.0269$**.
   - Fusing OOD features into ordinal regression degraded QWK ($\Delta = -0.0042$, $p = 0.4897$).
   - OOD features and uncertainty features are collinear ($r = -1.0000$).
   - Species addition (Contrast D vs C) is statistically significant ($p_{\text{adj}} = \mathbf{0.0115}, d_z = +0.7883$).
2. **The True Mechanism of Stabilization:**
   - Species conditioning prevents catastrophic false-alarm explosions on minority grades caused by extreme inverse class weighting ($cw_2 = 24.41\times$).
   - It supplies a categorical biological prior that prevents the model from predicting non-existent defects on specific fruit species (e.g., Grade 2 on Tomato or Mango).

---

## 4. Prioritized Execution Plan for Publication Remediation

Before any manuscript is submitted to IEEE Transactions, the following prioritized steps must be executed:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 5.0 REMEDIATION ROADMAP (MANDATORY)                    │
├──────┬──────────────────────────────────────────┬────────────────────────────────┤
│ Step │ Technical Action                         │ Target Deliverable             │
├──────┼──────────────────────────────────────────┼────────────────────────────────┤
│ 1    │ Deploy Corrected Split (split_v2)        │ master_dataset_v2.csv          │
│      │ Reallocate session 20200730 to test set  │ Guaranteed Grade 2 test N=119  │
├──────┼──────────────────────────────────────────┼────────────────────────────────┤
│ 2    │ Extract Real OOD Features                │ outputs/features/ood_real.pt   │
│      │ Run ConvNeXt on open-set-dataset         │ Cherry, Kiwi, Peach, Pineapple │
├──────┼──────────────────────────────────────────┼────────────────────────────────┤
│ 3    │ Execute Real-World Open-Set Benchmark    │ phase5_real_ood_benchmark.json │
│      │ Measure AUROC, AUPR, FPR95 on real OOD   │ Replace synthetic numbers      │
├──────┼──────────────────────────────────────────┼────────────────────────────────┤
│ 4    │ Adopt Effective Sample Weighting         │ beta=0.999 weighting           │
│      │ Bound weight ratio from 111:1 to 8.5:1   │ Eliminate Seed 19/31 collapse  │
├──────┼──────────────────────────────────────────┼────────────────────────────────┤
│ 5    │ Run Pre-Registered Confirmatory Study    │ N=30 seeds, Config B vs A      │
│      │ Save all checkpoints and raw predictions │ Full provenance persistence    │
├──────┼──────────────────────────────────────────┼────────────────────────────────┤
│ 6    │ Stratified Reporting by Domain           │ Report fruits360 vs real-world │
│      │ Leave-One-Domain-Out benchmark           │ Honest domain shift appraisal  │
└──────┴──────────────────────────────────────────┴────────────────────────────────┘
```

---

## 5. Audit Deliverables Verification Table

All 12 required deliverables for Phase 4.5 have been generated and verified on disk in `C:\fruit_dataset\outputs\casm_v2\`:

| Deliverable File | Verification Status | Primary Content |
|---|:---:|---|
| `phase45_experiment_provenance.csv` | **VERIFIED** | 172 experimental run records spanning Phases 1 through 4.4 |
| `phase45_provenance_audit.md` | **VERIFIED** | Forensic resolution of historical 0.8859 vs 0.7032 discrepancy |
| `phase45_dataset_integrity.json` | **VERIFIED** | Complete census: zero path leakage, zero content duplicate hashes |
| `phase45_dataset_integrity_report.md` | **VERIFIED** | Definitive proof of Grade 2 test absence and domain confounding |
| `phase45_model_implementation_audit.md` | **VERIFIED** | Parameter reconciliation (626k vs claimed 772); CORAL cutpoint analysis |
| `phase45_metric_reproduction.csv` | **VERIFIED** | Recomputed mean, std, median, min, max across 4 configs |
| `phase45_paired_statistics.csv` | **VERIFIED** | Recomputed paired $t$-test ($p=0.0593$), Wilcoxon ($p=0.0637$), Holm-Bonferroni |
| `phase45_ablation_reconciliation.csv` | **VERIFIED** | Recomputed all 6 factorial contrasts, CIs, and multiplicity control |
| `phase45_statistical_reproduction_report.md` | **VERIFIED** | Synthesis of statistical evidence, sensitivity analysis ($N=18$) |
| `phase45_stability_and_domain_report.md` | **VERIFIED** | Empirical proof of Seed 19/31 Grade 2 false-alarm explosion |
| `phase45_corrected_experimental_protocol.md` | **VERIFIED** | Pre-registered protocol for Phase 5.0 with versioned split and persistence |
| `phase45_final_research_decision.md` | **VERIFIED** | Concluding audit verdict: Outcome D with remediation roadmap |

Companion implementation files:
- `outputs/casm_v2/casm_v2_corrected_modules.py`: Monotonic CORAL head and robust species fallback.
- `outputs/casm_v2/test_corrected_modules.py`: 4 unit tests passing 100%.
- `outputs/casm_v2/phase45_stability_diagnostics.json`: Diagnostic histories, confusion matrices, and domain metrics.

---

## 6. Concluding Recommendation for Research Leadership

**DO NOT SUBMIT THE MANUSCRIPT IN ITS CURRENT STATE.**  
Submitting a paper claiming 5-grade ordinal grading on a test set containing 0 instances of Grade 2, accompanied by an OOD benchmark generated from synthetic Gaussian numbers, invites immediate, justified desk rejection or severe ethical criticism.

By executing the 6-step remediation plan (Step 1–6) using the corrected protocol and modules established in this audit, the research will be transformed into an unassailable, publication-grade contribution for *IEEE Transactions on Pattern Analysis and Machine Intelligence* or *IEEE Transactions on Cybernetics*.
