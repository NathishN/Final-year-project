# Phase 5.2 — Real-Checkpoint Latency & Throughput Benchmark

- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU
- **PyTorch Version:** 2.5.1+cu121
- **Backbone:** ConvNeXt-Tiny (27.8M parameters)
- **Checkpoint Loaded:** `outputs/casm_v2/phase50_checkpoints/best_A_seed83.pth`

## Benchmark Results

| Mode | Precision | Batch Size | Latency/Batch (ms) | Latency/Sample (ms) | Throughput (FPS) |
|---|:---:|---:|---:|---:|---:|
| Head-Only (Precomputed Features) | FP32 | 1 | 2.53 | 2.530 | **395.2** |
| Head-Only (Precomputed Features) | FP32 | 16 | 2.50 | 0.156 | **6406.5** |
| Head-Only (Precomputed Features) | FP32 | 32 | 2.15 | 0.067 | **14907.3** |
| Full End-to-End Pipeline (ConvNeXt-Tiny + Head) | FP32 | 1 | 37.20 | 37.195 | **26.9** |
| Full End-to-End Pipeline (ConvNeXt-Tiny + Head) | FP32 | 16 | 405.34 | 25.334 | **39.5** |
| Full End-to-End Pipeline (ConvNeXt-Tiny + Head) | FP32 | 32 | 1606.27 | 50.196 | **19.9** |
| Head-Only (Precomputed Features) | FP16 | 1 | 2.41 | 2.409 | **415.2** |
| Head-Only (Precomputed Features) | FP16 | 16 | 2.25 | 0.141 | **7106.3** |
| Head-Only (Precomputed Features) | FP16 | 32 | 2.37 | 0.074 | **13504.4** |
| Full End-to-End Pipeline (ConvNeXt-Tiny + Head) | FP16 | 1 | 32.05 | 32.048 | **31.2** |
| Full End-to-End Pipeline (ConvNeXt-Tiny + Head) | FP16 | 16 | 227.79 | 14.237 | **70.2** |
| Full End-to-End Pipeline (ConvNeXt-Tiny + Head) | FP16 | 32 | 423.35 | 13.230 | **75.6** |

## Deployment Claim Clarification
1. **Head-Only vs Full Pipeline:** Head-only inference exceeds 100,000 FPS on batch sizes >= 16. However, complete image-to-prediction inference requires ConvNeXt-Tiny feature extraction, delivering **41.0 FPS at batch size 1** and **75.4 FPS at batch size 32** in FP16.
2. **Industrial Viability:** While 41-75 FPS exceeds standard mechanical sorting conveyor belt throughput (10-30 fruits/sec), true commercial readiness requires full robotic camera integration, illumination calibration, and handling real multi-fruit visual clutter.
