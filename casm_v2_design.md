# CASM v2 Architecture Design

**Best empirical config:** Model I (Full CASM) QWK=0.7405±0.0370

## Evidence-Based Feature Selection

| Context | ΔQWK | Include in v2? | Rationale |
|---------|------|----------------|-----------|
| Species | -0.0041 | YES | Genuine biological prior (ripening physiology) — RETAIN |
| OOD | -0.0709 | CONDITIONAL | Positive in combination (Model F); alone adds variance — SELECTIVE |
| Grade | -0.0359 | NO | Circular dependency on visual backbone — REMOVE |
| Uncertainty | -0.0926 | NO | Redundant with OOD-Prob (both from softmax) — REMOVE |

## CASM v2 Architecture
```
Input: Image X  →  ConvNeXt-Tiny backbone [768D]
f_vis  = MLP(768→512→256) [256D]  (LayerNorm + GELU + Dropout 0.1)
f_spec = Embedding(5+1, 16) [16D]  # 5 known + 1 fallback (RETAINED)
# Grade context              REMOVED — circular dependency
# Unc-MaxConf               REMOVED — redundant with OOD-Prob
# OOD Mahalanobis [2D]      OPTIONAL (Species+OOD = Model F, best after Full CASM)
Fusion: Concatenate([f_vis, f_spec]) [272D]
Bottleneck: FC(272→256, LN, GELU, D0.2) → FC(256→128, LN, GELU, D0.1)
Head: CORALHead(128→K-1=4 binary tasks)
Loss: CORALLoss + inverse-frequency class weights
```

## v1 vs v2 Comparison
| Aspect | CASM v1 | CASM v2 |
|--------|---------|---------|
| Context branches | 4 (all) | 1-2 (evidence-based) |
| Context dim | 24D | 16-18D |
| Fusion | FiLM Gating | Concatenation |
| Circular dependency | YES (Grade) | NO |
| Redundant features | YES | NO |