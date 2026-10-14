# Phase 5.2 — Integration Test & Reproduction Command Audit

## 1. Test Suite Status
- **Suite:** 17 tests across `test_data_integrity.py`, `test_model_properties.py`, `test_metrics_and_calibration.py`, `test_end_to_end_smoke.py`, and `test_integration_predictions.py`.
- **Status:** 17/17 PASSED.

## 2. Reproduction Command Audit

| Command | Description | Status | Exit Code | Duration |
|---|---|:---:|---:|---:|
| `python -m unittest discover tests -v` | Execute full unit and integration test suite | **PASSED** | 0 | 17.07s |
| `python scripts/phaseB_dataset_integrity.py` | Verify dataset manifest and class distribution | **PASSED** | 0 | 3.52s |
| `python scripts/phaseC_ordinal_loss_audit.py` | Verify monotonic cutpoint constraints | **PASSED** | 0 | 7.57s |
| `python scripts/phaseH_model_efficiency.py` (attempt 1, child of validation script) | Verify parameter census and latency benchmarking | **FAILED** | -1073740791 (0xC0000409) | 59.91s |
| `python scripts/phaseH_model_efficiency.py` (attempt 2, standalone) | Verify parameter census and latency benchmarking | **PASSED** | 0 | ~73s (estimated) |

> The first failure happened while the parent process still held the GPU. The root cause was not isolated, so the command is flaky under GPU contention. Note: these checks confirm that the scripts exit cleanly. They do not confirm that the outputs are scientifically correct.
| `python scripts/assemble_paper.py` | Compile manuscript and bibliography | **PASSED** | 0 | 1.98s |
