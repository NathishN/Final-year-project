# Model Architecture, Parameter Census & Efficiency Audit Report

**Date:** October 2026  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) & x86_64 CPU  

---

## 1. Parameter Census Breakdown

| Component | Architecture | Trainable Parameters | Total Parameters | Checkpoint Size | Role |
|:---|:---|---:|---:|---:|:---|
| **ConvNeXt-Tiny Backbone** | 4-Stage ConvNet (768-D) | 0 (Frozen) | **27,823,973** | ~114 MB | Visual representation extraction |
| **Config A Head (Image Only)** | 2-Layer MLP + MonotonicCORAL | 626,180 | **626,180** | ~2.5 MB | Optimal configuration |
| **Config B Head (Species)** | MLP + Species Emb (16-D) | 630,372 | **630,372** | ~2.5 MB | Aux species conditioning |
| **Config D Head (Full CASM)** | Multi-Modal Fusion | 630,884 | **630,884** | ~2.5 MB | Species + OOD conditioning |

> **Parameter Efficiency Finding:** The lightweight ordinal heads introduce only **~626,000 to ~630,000 parameters** (~2.1% of total parameters). The ConvNeXt-Tiny visual backbone accounts for 97.9% of model size (~28.6M parameters) and remains frozen during ordinal training.

---

## 2. Latency and Throughput Benchmarks

### A. Head-Only Inference (Precomputed Features -> Monotonic CORAL Head)

| Compute Device | Batch Size | Total Batch Latency (ms) | Latency per Image (ms) | Throughput (FPS) |
|:---|---:|---:|---:|---:|
| **CPU** | 1 | 1.108 ms | **1.108 ms** | **902.5 FPS** |
| **CPU** | 16 | 1.316 ms | **0.0822 ms** | **12,158.8 FPS** |
| **CPU** | 64 | 1.69 ms | **0.0264 ms** | **37,876.7 FPS** |
| **CPU** | 256 | 3.248 ms | **0.0127 ms** | **78,816.7 FPS** |
| **CPU** | 2048 | 18.859 ms | **0.0092 ms** | **108,593.4 FPS** |
| **NVIDIA RTX 3050 GPU** | 1 | 1.906 ms | **1.9062 ms** | **524.6 FPS** |
| **NVIDIA RTX 3050 GPU** | 16 | 2.155 ms | **0.1347 ms** | **7,423.1 FPS** |
| **NVIDIA RTX 3050 GPU** | 64 | 1.984 ms | **0.031 ms** | **32,251.2 FPS** |
| **NVIDIA RTX 3050 GPU** | 256 | 2.119 ms | **0.0083 ms** | **120,818.5 FPS** |
| **NVIDIA RTX 3050 GPU** | 2048 | 6.772 ms | **0.0033 ms** | **302,417.9 FPS** |

### B. Full End-to-End Pipeline Inference (Raw Image -> ConvNeXt-Tiny -> Head)

| Compute Device | Batch Size | Total Batch Latency (ms) | Latency per Image (ms) | Throughput (FPS) | Industrial Suitability |
|:---|---:|---:|---:|---:|:---|
| **NVIDIA RTX 3050 GPU (FP16)** | 1 | 24.386 ms | **24.3858 ms** | **41.0 FPS** | Real-Time Sorting Conveyor (>30 FPS) |
| **NVIDIA RTX 3050 GPU (FP16)** | 8 | 115.532 ms | **14.4415 ms** | **69.2 FPS** | Real-Time Sorting Conveyor (>30 FPS) |
| **NVIDIA RTX 3050 GPU (FP16)** | 16 | 214.5 ms | **13.4062 ms** | **74.6 FPS** | Real-Time Sorting Conveyor (>30 FPS) |
| **NVIDIA RTX 3050 GPU (FP16)** | 32 | 420.171 ms | **13.1303 ms** | **76.2 FPS** | Real-Time Sorting Conveyor (>30 FPS) |

> **Throughput Claim Clarification:** Historical claims of '>100,000 FPS' reflect head-only processing on pre-extracted feature tensors at large batch sizes. Full end-to-end processing (image ingestion, ConvNeXt-Tiny feature extraction, and Monotonic CORAL classification) achieves **39.3 to 76.1 FPS** in FP16 on the RTX 3050 GPU, fully satisfying commercial conveyor sorting rates (typically 10–30 items/sec per sorting lane).

---

## 3. Deployment Safety Features in `inference_pipeline.py`

1. **Monotonic Probability Decoupling:** Reconstructs discrete probabilities $P(Y=k) \ge 0$ summing to 1.0 unconditionally.
2. **Confidence-Gated Abstention:** Triggers `ROUTE_TO_MANUAL_INSPECTION` whenever maximum confidence falls below 70%.
3. **Unknown Species Fallback:** Automatically clamps unknown species queries to index 5 without throwing exceptions.
4. **FP16 Mixed Precision Safety:** Validated against extreme activations without underflow or overflow.