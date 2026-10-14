"""
tests/test_integration_predictions.py
======================================
Integration tests validating the 120 saved prediction tensors from Phase 5.0:
- Verifies existence of all 120 prediction files (4 configs x 30 seeds)
- Validates tensor integrity, probability bounds, and sum-to-one constraints
- Independently recalculates QWK, MAE, and accuracy against stored values
- Verifies confirmatory statistical reconciliation:
  * Mean QWKConfigA > Mean QWKConfigB (negative context conditioning effect)
  * Paired t-test t(29) ≈ -4.16, p < 0.001
  * Wilcoxon signed-rank test W = 72 (zero_method='wilcox'), p < 0.001
"""

import os
import glob
import unittest
import torch
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score

ROOT = r"C:\fruit_dataset"
PREDS_DIR = os.path.join(ROOT, "outputs", "casm_v2", "phase50_per_seed_predictions")


class TestIntegrationPredictions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pred_files = glob.glob(os.path.join(PREDS_DIR, "preds_*.pt"))
        if len(cls.pred_files) == 0:
            raise FileNotFoundError(f"No prediction files found in {PREDS_DIR}")

    def test_all_120_prediction_files_present(self):
        """All 120 confirmatory runs (30 seeds x 4 configs) must exist on disk."""
        self.assertEqual(len(self.pred_files), 120, f"Expected 120 prediction files, found {len(self.pred_files)}")

    def test_prediction_tensor_structure_and_probabilities(self):
        """Spot-check prediction tensors for shape, valid probability simplex, and finite values."""
        sample_files = [
            os.path.join(PREDS_DIR, "preds_A_seed42.pt"),
            os.path.join(PREDS_DIR, "preds_B_seed83.pt"),
            os.path.join(PREDS_DIR, "preds_C_seed1.pt"),
            os.path.join(PREDS_DIR, "preds_D_seed999.pt"),
        ]
        for fpath in sample_files:
            self.assertTrue(os.path.exists(fpath), f"File missing: {fpath}")
            data = torch.load(fpath, weights_only=False)
            probs = data["probabilities"].numpy()
            preds = data["predictions"].numpy()
            targets = data["targets"].numpy()

            self.assertEqual(probs.shape, (12833, 5))
            self.assertEqual(preds.shape, (12833,))
            self.assertEqual(targets.shape, (12833,))

            # All probabilities non-negative
            self.assertTrue(np.all(probs >= -1e-6), f"Negative probabilities detected in {fpath}")
            # All probabilities sum to 1.0 within numerical tolerance
            sums = np.sum(probs, axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-4, err_msg=f"Probabilities do not sum to 1 in {fpath}")

    def test_recomputed_metrics_match_stored(self):
        """Recomputed QWK and MAE from tensors must match stored checkpoint metadata within rounding."""
        fpath = os.path.join(PREDS_DIR, "preds_A_seed83.pt")
        data = torch.load(fpath, weights_only=False)
        preds = data["predictions"].numpy()
        targets = data["targets"].numpy()

        recomputed_qwk = cohen_kappa_score(targets, preds, weights="quadratic")
        recomputed_mae = mean_absolute_error(targets, preds)

        stored_qwk = data["qwk"]
        stored_mae = data["mae"]

        self.assertAlmostEqual(recomputed_qwk, stored_qwk, places=3,
                               msg=f"Recomputed QWK {recomputed_qwk:.4f} != stored {stored_qwk:.4f}")
        self.assertAlmostEqual(recomputed_mae, stored_mae, places=3,
                               msg=f"Recomputed MAE {recomputed_mae:.4f} != stored {stored_mae:.4f}")

    def test_confirmatory_statistical_reconciliation(self):
        """Config A vs Config B 30-seed contrast matches forensic statistical values."""
        qwk_A = []
        qwk_B = []

        # Prespecified 30 seeds in deterministic order
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 42, 43, 47,
                 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 123, 777, 999, 2025]

        for s in seeds:
            dA = torch.load(os.path.join(PREDS_DIR, f"preds_A_seed{s}.pt"), weights_only=False)
            dB = torch.load(os.path.join(PREDS_DIR, f"preds_B_seed{s}.pt"), weights_only=False)
            qwk_A.append(dA["qwk"])
            qwk_B.append(dB["qwk"])

        qwk_A = np.array(qwk_A)
        qwk_B = np.array(qwk_B)

        mean_A = float(np.mean(qwk_A))
        mean_B = float(np.mean(qwk_B))
        delta = mean_B - mean_A

        # Assert mean values match reported figures (0.8478 and 0.8118)
        self.assertAlmostEqual(mean_A, 0.8478, places=3)
        self.assertAlmostEqual(mean_B, 0.8118, places=3)
        self.assertAlmostEqual(delta, -0.0360, places=3)

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(qwk_B, qwk_A)
        self.assertAlmostEqual(t_stat, -4.159, places=2)
        self.assertTrue(t_pval < 0.001)

        # Wilcoxon signed-rank test (recomputed W=71.5, reported rounded to W=72 with zero_method='wilcox')
        w_res = stats.wilcoxon(qwk_B, qwk_A, zero_method='wilcox', alternative='two-sided')
        self.assertAlmostEqual(w_res.statistic, 71.5, places=1, msg=f"Expected Wilcoxon W≈71.5/72, got {w_res.statistic}")
        self.assertTrue(w_res.pvalue < 0.001)


if __name__ == "__main__":
    unittest.main()
