# Phase 4.3 Context Contribution Audit — Master Report

**Project:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading
**Experiment:** 9 models × 5 seeds = 45 training runs | Dataset: 56,036 samples

## Executive Summary

> Full CASM ΔQWK=+0.0183 vs Image-Only (positive trend, p=0.4445, NS).
> Best config: Model I (Full CASM) QWK=0.7405±0.0370
> Recommendation: CASM v2 with Species-Aware concatenation fusion.

## Full Results Table
| Model | Config | Acc Mean±Std | MAE Mean±Std | QWK Mean±Std | ΔQWK |
|-------|--------|-------------|-------------|-------------|------|
| **A** | Image Only | 87.06±0.65% | 0.2276±0.0061 | 0.7222±0.0210 | +0.0000 |
| **B** | Image + Species | 85.86±0.74% | 0.2542±0.0166 | 0.7181±0.0188 | -0.0041 |
| **C** | Image + OOD | 86.18±2.72% | 0.2522±0.0659 | 0.6513±0.0963 | -0.0709 |
| **D** | Image + Grade | 85.53±3.85% | 0.2532±0.0740 | 0.6863±0.1016 | -0.0359 |
| **E** | Image + Uncertainty | 86.39±1.94% | 0.2653±0.0852 | 0.6296±0.1912 | -0.0926 |
| **F** | Image + Species + OOD | 87.52±0.82% | 0.2156±0.0193 | 0.7385±0.0200 | +0.0163 |
| **G** | Image + Species + Grade | 86.31±1.33% | 0.2411±0.0357 | 0.7313±0.0402 | +0.0091 |
| **H** | Image + OOD + Grade | 86.65±1.38% | 0.2191±0.0248 | 0.7106±0.0327 | -0.0116 |
| **I** ★ | Full CASM | 87.93±0.80% | 0.2075±0.0321 | 0.7405±0.0370 | +0.0183 |

## Context Branch Marginal Impact
| Branch | ΔQWK | Verdict |
|--------|------|---------|
| Species | -0.0041 | NEUTRAL |
| OOD | -0.0709 | HARMFUL |
| Grade | -0.0359 | HARMFUL |
| Uncertainty | -0.0926 | HARMFUL |

## Claim Verdicts
| Claim | Context | ΔQWK | Verdict |
|-------|---------|------|---------|
| Claim A | Full Context | +0.0183 | NOT SUPPORTED (Neutral) |
| Claim B | OOD | -0.0709 | NOT SUPPORTED (Harmful) |
| Claim C | Species | -0.0041 | NOT SUPPORTED (Neutral) |
| Claim D | Grade | -0.0359 | NOT SUPPORTED (Harmful) |
| Claim E | Uncertainty | -0.0926 | NOT SUPPORTED (Harmful) |
| Claim F | Species+OOD | +0.0163 | NOT SUPPORTED (Neutral) |
| Claim G | Species+Grade | +0.0091 | NOT SUPPORTED (Neutral) |

## Deliverables
| # | File | Status |
|---|------|--------|
| 1 | context_contribution_results.csv | ✅ |
| 2 | context_significance_report.md   | ✅ |
| 3 | context_noise_analysis.md        | ✅ |
| 4 | context_importance_report.md     | ✅ |
| 5 | architecture_failure_report.md   | ✅ |
| 6 | casm_v2_design.md               | ✅ |
| 7 | research_claim_validation.md     | ✅ |
| 8 | title_revision_report.md        | ✅ |
| 9 | phase4_context_audit_review.md  | ✅ |
| 10| phase4_context_audit_master_report.md | ✅ |

## IEEE Recommendation
**MAJOR REVISION REQUIRED → ACCEPT AFTER REVISION**
1. Replace Context-Aware → Species-Aware in title
2. Implement CASM v2 (Species-Aware concatenation)
3. Remove circular Grade context and redundant Unc-MaxConf
4. Present Context Contribution Audit as methodological contribution