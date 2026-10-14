# IEEE Transactions Peer Review — Context Contribution Audit

**Manuscript:** Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading...
**Recommendation:** MAJOR REVISION REQUIRED → ACCEPT AFTER REVISION

## Performance Summary
- Image-Only (A): QWK=0.7222±0.0210  Acc=87.06%
- Full CASM  (I): QWK=0.7405±0.0370  Acc=87.93%
- Gap: ΔQWK=+0.0183 (IMPROVEMENT, p=0.4445, NS)
- Best: Model I (Full CASM) QWK=0.7405

## Major Strengths
1. Multi-phase pipeline (Phases 1-4) well-structured and reproducible
2. 9-model × 5-seed ablation with paired t-tests — exemplary design
3. CORAL ordinal regression mathematically correct for severity ordering
4. Scientific honesty in reporting the central contradiction
5. Novel Context Contribution Audit methodology is itself a contribution

## Required Revisions
W1 [CRITICAL]: Replace 'Context-Aware' with 'Species-Aware' in title
W2 [HIGH]: Remove circular Grade context (derived from same visual features)
W3 [HIGH]: Remove Unc-MaxConf (redundant with OOD-Prob from softmax)
W4 [MEDIUM]: Replace FiLM gating with concatenation fusion
W5 [MEDIUM]: Report Cohen's d effect sizes alongside p-values

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

## Final Verdict
MAJOR REVISION REQUIRED. Phases 1-3 are strong. Phase 4 requires honest reframing.
Context Contribution Audit is a publishable negative finding with methodological value.
If CASM v2 (Species-Aware) validates the improvement: ACCEPT AFTER MAJOR REVISION.