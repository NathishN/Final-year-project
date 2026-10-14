# OOD Metric Reconciliation and Confounding Verification

## Top-Level AUROC Consistency (metadata vs. reported)

| Detector | Metadata AUROC | Reported AUROC | Difference |
|----------|---------------|---------------|-----------|
| Mahalanobis | 84.22 | 84.22 | 0.0 |
| Energy | 81.99 | 81.99 | 0.0 |
| MSP | 74.26 | 74.26 | 0.0 |

All top-level values match between features_metadata.json and reported values.

## Per-Domain Analysis — CRITICAL FINDINGS

| Domain | n | Mahal AUROC | Energy AUROC | MSP AUROC |
|--------|---|-------------|--------------|-----------|
| fruits360 | 10974 | 93.26 | 79.99 | 75.21 |
| real_grade | 67 | 18.81 | 98.89 | 96.57 |
| real_quality | 1480 | 18.08 | 92.49 | 61.74 |
| real_severity | 312 | 94.1 | 98.67 | 95.3 |

## Key Discrepancy: The OOD Domain Collapse Claim Is Detector-Specific

The prior Phase 5.0 report states:
> 'Mahalanobis AUROC inverts to 18.08% on market bins (real_quality), proving the detector
> relies on white background features rather than pure fruit morphology.'

**Verified finding from metadata:**
- Mahalanobis: 18.08% on real_quality — CONFIRMED inversion vs. fruits360 (93.26%)
- BUT Energy AUROC on real_quality: 92.49% — NOT inverted
- AND MSP AUROC on real_quality: 61.74% — NOT inverted
- AND Energy/MSP on real_grade: 98.89% / 96.57% — actually EXCELLENT

**The prior report conflates Mahalanobis-specific failure with universal OOD failure.**
Only Mahalanobis shows the inversion pattern. Energy and MSP perform differently.

## Independent Recomputation Status

- **Cannot independently recompute AUROC** — OOD predictions CSV lacks in-distribution scores.
  The CSV (ood_predictions.csv) contains only OOD samples (is_ood=1 for all 5,000 rows).
- **Feature file contains OOD logits** — MSP and Energy scores computed and verified against stored logits.
- **Status:** PARTIALLY_VERIFIED (metadata-consistent; not independently computed from score vectors).

## Causal Background Claim

CLAIM: 'Background style (white studio vs. market clutter) causes OOD failure'
STATUS: UNRESOLVED
EVIDENCE: Mahalanobis failure correlation only. No foreground-masked experiment performed.
NEXT ACTION: Run GrabCut/SAM segmentation experiment on real_quality images.