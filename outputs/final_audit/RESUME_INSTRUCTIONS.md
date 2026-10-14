# Master Audit — Resume & Recovery Instructions

If execution is interrupted at any point, follow these instructions to resume:

## 1. Environment & Hardware Verification
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Check Existing Completed Artifacts
- Checkpoints: `C:\fruit_dataset\outputs\casm_v2\phase50_checkpoints\` (120 models)
- Raw Predictions: `C:\fruit_dataset\outputs\casm_v2\phase50_per_seed_predictions\` (120 `.pt` tensors)
- Real OOD Features: `C:\fruit_dataset\outputs\casm_v2\phase50_ood_features\ood_real_features.pt`
- Master Dataset: `C:\fruit_dataset\master_dataset.csv`
- Split Manifest: `C:\fruit_dataset\outputs\casm_v2\phase50_split_v2_manifest.csv`

## 3. Resume Commands by Phase
- **Phase A (Inventory):** `python scripts/audit_repository_inventory.py`
- **Phase B (Dataset & Leakage):** `python scripts/audit_dataset_leakage.py`
- **Phase C (Ordinal & Loss):** `python scripts/audit_ordinal_head.py`
- **Phase D (Confirmatory Analysis):** `python scripts/audit_confirmatory_reproduction.py`
- **Phase E (Real-Image OOD):** `python scripts/audit_ood_validity.py`
- **Phase F (Domain Generalization):** `python scripts/audit_domain_generalization.py`
- **Phase G (Calibration):** `python scripts/audit_calibration.py`
- **Phase H (Efficiency & Inference API):** `python scripts/audit_model_efficiency.py`
- **Phase I (Automated Test Suite):** `pytest tests/`
- **Phase K (Claim Evidence Matrix):** `python scripts/generate_claim_matrix.py`
- **Phase L (Paper Package):** `python scripts/assemble_paper.py`
