# Manuscript Revision Log — Phase 5.1 Forensic Audit

**Created:** 2026-10-13  
**Basis:** Evidence from `outputs/phase51_forensic_audit/FINAL_FORENSIC_REPORT.md`

---

## Revision 1: Training Protocol Disclosure (HIGH PRIORITY)

**Section affected:** `methodology.md` §4, `manuscript.md` Abstract and Methodology

**Original text (§4 opening):**
> "Configuration A (Image Only): Frozen ConvNeXt-Tiny embeddings (768-D) → ..."

**Problem:** The word "Frozen" is present in the architecture description, but there is no explicit statement that:
1. Training operates on **precomputed feature tensors** (`phase4_dataset.pt`), not raw images
2. The backbone is not instantiated during training at all
3. Training batch size (2048) is feature-batch size, not image-batch size

**Required addition to Methodology §4:**

```
### Feature Extraction and Frozen Backbone Protocol

All 120 confirmatory experiments operate on precomputed ConvNeXt-Tiny feature tensors 
extracted once and cached in `outputs/features/phase4_dataset.pt` (56,036 samples × 768 dimensions, 
179 MB). The ConvNeXt-Tiny backbone (~28.6M parameters) is not instantiated during training; 
no gradient updates flow through the visual encoder. 

Only the Monotonic CORAL head and fusion MLP (~626K–630K parameters depending on configuration) 
receive gradient updates during training (AdamW, lr=1e-3, weight_decay=1e-4, 15 epochs, 
batch_size=2048 feature vectors).

This protocol was chosen to: (a) ensure fair comparison across configurations with identical 
visual representations, (b) reduce training cost for the 120-run confirmatory study on 
consumer GPU hardware (RTX 3050 6GB), and (c) isolate the effect of context conditioning 
from backbone initialization variance.
```

---

## Revision 2: Statistical Correction — Wilcoxon W (HIGH PRIORITY)

**Section affected:** `manuscript.md` Results, `abstract.md`

**Original text (line 15 of manuscript.md):**
> "paired t(29) = -4.16, p = 0.000259"

**Correction required:** The Wilcoxon statistic was reported inconsistently as W=59 (Phase 4.4) and W=62 (Phase 5.0). The correct value, independently recomputed from saved prediction tensors, is:
- **W = 72** (`scipy.stats.wilcoxon`, `zero_method='wilcox'`, `alternative='two-sided'`)
- **p = 0.000555**

The corrected Wilcoxon result still strongly rejects H₀ — the directional conclusion is unchanged. The t-test result (p=0.000259) is confirmed correct.

**Updated results text:**
```
Wilcoxon signed-rank test: W = 72 (zero_method='wilcox'), p = 0.000555
Paired t-test: t(29) = -4.16, p = 0.000260 [confirmed]
```

---

## Revision 3: OOD Domain Collapse Claim (HIGH PRIORITY)

**Section affected:** `manuscript.md` Abstract (line 17), Results, `abstract.md`

**Original text:**
> "Mahalanobis Distance achieves 84.22% AUROC within studio domains, but drops to 18.08% on 
> market-bin images, exposing severe acquisition background confounding."

**Problems identified by Phase 5.1 audit:**
1. The claim attributes the failure to "acquisition background confounding" — this is a causal 
   claim without controlled experimental evidence (no foreground segmentation experiment was run).
2. The domain collapse is **Mahalanobis-specific**: Energy AUROC on real_quality is 92.49%, 
   MSP AUROC is 61.74% — neither shows a comparable inversion.
3. On real_grade: Energy AUROC = 98.89%, MSP AUROC = 96.57% — these detectors actually excel.

**Required corrections:**

Change:
> "exposing severe acquisition background confounding"

To:
> "suggesting that Mahalanobis distance detection may rely on studio acquisition artifacts 
> (white background) rather than fruit morphology, though causal attribution requires 
> foreground-masked evaluation (not conducted in this study). Energy and MSP detectors 
> do not show comparable domain-specific failure."

Add table to OOD results:

| Domain | n | Mahalanobis | Energy | MSP |
|--------|---|------------|--------|-----|
| fruits360 (studio) | 10,974 | 93.26% | 79.99% | 75.21% |
| real_severity | 312 | 94.10% | 98.67% | 95.30% |
| real_quality (market) | 1,480 | **18.08%** | 92.49% | 61.74% |
| real_grade | 67 | **18.81%** | **98.89%** | **96.57%** |

---

## Revision 4: Throughput / Speed Claim (HIGH PRIORITY)

**Section affected:** `manuscript.md` Architecture/Efficiency claims, `README.md`

**Original claim:**
> ">100,000 FPS on RTX 3050 GPU"

**Problem:** This measurement applies ONLY to the Monotonic CORAL head inferencing on 
precomputed 768-dimensional feature vectors. The full end-to-end pipeline (backbone image 
preprocessing + ConvNeXt-Tiny feature extraction + CORAL head) throughput was NOT benchmarked.

**Required correction:**

```
## Inference Efficiency

Head-only throughput (precomputed ConvNeXt-Tiny features → CORAL head, GPU batch=2048): 
>120,000 samples/sec (benchmarked on RTX 3050 6GB).

Full end-to-end throughput (raw image → backbone → head): not independently benchmarked 
in this study. For reference, ConvNeXt-Tiny achieves approximately 200-600 images/sec 
on consumer GPUs at typical batch sizes.

Trainable parameters: 626,180 (Config A) – 630,884 (Config D). 
Total model parameters (backbone + head): ~29.2M (backbone is frozen at inference).
```

---

## Revision 5: Session Exclusivity Qualification (MEDIUM PRIORITY)

**Section affected:** `methodology.md` dataset section, `paper/results.md`

**Original claim:**
> "Session `20200730` (n=119 Banana Grade 2 images) moved into test"

**Correction:** Session 20200730 contains two subgroups:
- 119 Banana Grade 2 images → assigned to test ✓
- 147 Apple Grade 0 images → remain in train

Full session holdout was NOT enforced. The split enforces **within-species, within-grade 
session exclusivity** for the Grade 2 Banana subset. This is a reasonable but weaker 
guarantee than full session holdout. The manuscript should state:

```
"To prevent data leakage from the Grade 2 test set, all 119 Banana Grade 2 images from 
acquisition session 20200730 are exclusively assigned to the test split. Images from 
the same session in other species-grade combinations (Apple Grade 0, n=147) remain 
in the training partition. Full session-level holdout was not applied as session 
boundaries do not align with species boundaries in this dataset."
```

---

## Revision 6: ECE Qualification (MEDIUM PRIORITY)

**Section affected:** `manuscript.md` Abstract and Calibration results

**Original:**
> "Expected Calibration Error of 2.51%"

**Problem:** No seed was specified. Independent recomputation yields ECE=1.27% for seed 83 
(highest-QWK Config A seed). ECE varies across seeds. Additionally, pooled ECE is 
dominated by fruits360 studio images (85.5% of test set).

**Required addition:**
```
"Expected Calibration Error (ECE, 10 equal-mass bins, Config A mean over seeds): reported 
as 2.51% in prior analysis. Note: ECE varies by seed; domain-stratified ECE is not 
reported as in-distribution test data is dominated by studio images (85.5%). 
Per-domain calibration is a recommended direction for future work."
```

---

## Revision 7: README Reproduction Commands (HIGH PRIORITY)

**Section affected:** `README.md`

Replace:
```bash
python scripts/phaseB_dataset_audit.py      # → wrong name
python scripts/phaseC_ordinal_audit.py      # → wrong name
python scripts/phaseH_efficiency_audit.py   # → wrong name
```

With:
```bash
python scripts/phaseB_dataset_integrity.py    # Dataset integrity & split audit
python scripts/phaseC_ordinal_loss_audit.py   # Ordinal head validation
python scripts/phaseD_confirmatory_audit.py   # Recompute confirmatory metrics
python scripts/phaseE_ood_audit.py            # OOD evaluation
python scripts/phaseF_domain_audit.py         # Domain generalization audit
python scripts/phaseG_calibration_audit.py    # Calibration & selective prediction
python scripts/phaseH_model_efficiency.py     # Efficiency benchmarks
```

---

## Summary of Changes Required

| # | File | Type | Priority | Status |
|---|------|------|----------|--------|
| 1 | methodology.md | ADD: frozen feature protocol | HIGH | REQUIRED |
| 2 | manuscript.md + abstract.md | FIX: Wilcoxon W=62→W=72 | HIGH | REQUIRED |
| 3 | manuscript.md + abstract.md | FIX: OOD detector-specific claim | HIGH | REQUIRED |
| 4 | manuscript.md + README | FIX: throughput = head-only | HIGH | REQUIRED |
| 5 | methodology.md | QUALIFY: session exclusivity | MEDIUM | REQUIRED |
| 6 | manuscript.md | QUALIFY: ECE claim | MEDIUM | REQUIRED |
| 7 | README.md | FIX: script names | HIGH | REQUIRED |
