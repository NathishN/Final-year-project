# Forensic End-to-End Inference & Latency Audit

**Audit Timestamp:** 2026-10-14 10:22:57  

**Benchmarking Device:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) / Intel Core i5/i7  

**PyTorch Stack:** 2.5.1+cu121  

---

## 1. Reconciled Inference Benchmarks (Head-Only vs Full Pipeline)

| Pipeline Scope | Precision | Batch Size | Latency/Batch (ms) | Latency/Sample (ms) | Throughput (FPS) | Operational Meaning |
|---|:---:|---:|---:|---:|---:|---|
| Head-Only | FP32 | 1 | 2.53 | 2.530 | **395.2** | Feature-level only (Unrealistic) |
| Head-Only | FP32 | 16 | 2.50 | 0.156 | **6406.5** | Feature-level only (Unrealistic) |
| Head-Only | FP32 | 32 | 2.15 | 0.067 | **14907.3** | Feature-level only (Unrealistic) |
| Full Pipeline (E2E) | FP32 | 1 | 37.20 | 37.195 | **26.9** | Real Image to Grade (Deployable) |
| Full Pipeline (E2E) | FP32 | 16 | 405.34 | 25.334 | **39.5** | Real Image to Grade (Deployable) |
| Full Pipeline (E2E) | FP32 | 32 | 1606.27 | 50.196 | **19.9** | Real Image to Grade (Deployable) |
| Head-Only | FP16 | 1 | 2.41 | 2.409 | **415.2** | Feature-level only (Unrealistic) |
| Head-Only | FP16 | 16 | 2.25 | 0.141 | **7106.3** | Feature-level only (Unrealistic) |
| Head-Only | FP16 | 32 | 2.37 | 0.074 | **13504.4** | Feature-level only (Unrealistic) |
| Full Pipeline (E2E) | FP16 | 1 | 32.05 | 32.048 | **31.2** | Real Image to Grade (Deployable) |
| Full Pipeline (E2E) | FP16 | 16 | 227.79 | 14.237 | **70.2** | Real Image to Grade (Deployable) |
| Full Pipeline (E2E) | FP16 | 32 | 423.35 | 13.230 | **75.6** | Real Image to Grade (Deployable) |


## 2. Resolving the '>100,000 FPS' Throughput Discrepancy

Earlier project notes cited throughput figures exceeding **100,000 FPS** (and up to 302,417 FPS at batch size 2048):

- **Root Cause:** That benchmark strictly measured the 3-layer MLP projection and ordinal linear head (`~626k` parameters) when fed precomputed 768-D float32 tensors already in GPU VRAM.

- **Reality Check:** In any real-world agricultural packing house, raw RGB camera frames must pass through the 27.8M-parameter ConvNeXt-Tiny convolutional feature extractor.

- **Verified End-to-End Throughput:**

  - **Single image streaming (Batch Size 1, FP16):** 32.05 ms latency -> **31.2 FPS**

  - **Batched processing (Batch Size 16, FP16):** 14.24 ms/sample -> **70.2 FPS**

  - **Batched processing (Batch Size 32, FP16):** 13.23 ms/sample -> **75.6 FPS**


## 3. Industrial Grading Line Feasibility Assessment

- **Conveyor Belt Sorting Standard:** Standard optical grading conveyors transport fruit at **10 to 30 items per second** per sorting lane.

- **Pipeline Headroom:** At **31.2 FPS** (single-frame batch size 1) and **75.6 FPS** (batch size 32) in FP16 on edge-tier hardware (RTX 3050 Laptop GPU, 6GB VRAM), the architecture comfortably outpaces mechanical conveyor speeds.

- **Deployment Caveats:** True commercial grading requires accounting for camera frame grab latency, lighting fluctuations, multi-fruit occlusions, and edge device thermal throttling under continuous 24/7 sorting line duty.

