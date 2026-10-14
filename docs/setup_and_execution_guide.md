# Setup, Data Preparation, and Experiment Execution Guide

This comprehensive guide documents environment installation, data splitting, training, evaluation, troubleshooting, and reproducibility instructions for the project.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- PyTorch 2.0+ with CUDA support (tested on CUDA 12.x, NVIDIA RTX 3050 Laptop GPU)
- 16 GB System RAM

### Installation
```bash
pip install torch torchvision timm scikit-learn pandas numpy scipy pillow pytest
```

---

## 2. Dataset Preparation & Split Verification

The master dataset contains 56,036 active image samples across 5 species (*Apple, Banana, Mango, Orange, Tomato*).
To verify the dataset manifest and validate zero cross-split leakage:

```bash
python scripts/phaseB_dataset_integrity.py
```
Outputs:
- Manifest: `outputs/final_audit/split_manifest.csv`
- Leakage Check: `outputs/final_audit/split_integrity.csv`
- Report: `outputs/final_audit/dataset_integrity_report.md`

---

## 3. Automated Test Suite

To run all unit tests, property tests, and end-to-end smoke tests:

```bash
python -m unittest discover tests -v
```
Expected output:
```
Ran 17 tests in ~5s
OK
```

---

## 4. Confirmatory 30-Seed Training (4 Configurations)

To execute all 120 confirmatory runs under the frozen `split_v2` protocol with Effective Number of Samples weighting:

```bash
python phase50_stage5_confirmatory.py
```
Artifacts produced:
- 120 Checkpoints: `outputs/casm_v2/phase50_checkpoints/`
- 120 Prediction Tensors: `outputs/casm_v2/phase50_per_seed_predictions/`
- Result Table: `outputs/casm_v2/phase50_confirmatory_results.csv`
- Statistics Table: `outputs/casm_v2/phase50_confirmatory_statistics.csv`

---

## 5. Real-Image Open-Set (OOD) Benchmark

To extract ConvNeXt-Tiny embeddings on 5,000 real images of unseen fruit species and evaluate detectors:

```bash
python scripts/phaseE_ood_audit.py
```
Outputs:
- OOD Predictions: `outputs/final_audit/ood_predictions.csv`
- Per-Species Table: `outputs/final_audit/ood_per_species.csv`
- Validity Report: `outputs/final_audit/ood_validity_report.md`

---

## 6. Calibration & Selective Prediction Analysis

To evaluate Expected Calibration Error (ECE), Brier score, and selective risk-coverage curves:

```bash
python scripts/phaseG_calibration_audit.py
```
Outputs:
- Calibration Metrics: `outputs/final_audit/calibration_metrics.csv`
- Risk-Coverage Table: `outputs/final_audit/selective_risk_coverage.csv`
- Report: `outputs/final_audit/calibration_selective_prediction_report.md`

---

## 7. Model Efficiency & Production Inference Pipeline

To benchmark throughput, latency, and instantiate the production inference API:

```bash
python scripts/phaseH_model_efficiency.py
```

### Python Inference Example
```python
from scripts.inference_pipeline import FruitGrader

grader = FruitGrader()
result = grader.predict_image("path/to/fruit.jpg", confidence_threshold=0.70)
print(result)
```

---

## 8. Troubleshooting & Common Issues

| Issue | Root Cause | Solution |
|:---|:---|:---|
| Windows CP1252 Unicode Error | Terminal default stdout encoding is cp1252 | Run scripts with `python -X utf8 script_name.py` |
| PyTorch Out of Memory (OOM) | Batch size too large for 4-6 GB VRAM | Features are pre-extracted; set batch size to 1,024 or 2,048 |
| QWK is 0.0 on fruits360 | True labels are all Grade 0 (variance=0) | Mathematically expected; marked as `UNDEFINED (Constant Label)` |
