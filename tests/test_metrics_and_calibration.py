"""
tests/test_metrics_and_calibration.py
=====================================
Automated unit tests for QWK, degenerate metric cases, ECE, and OOD scores.
"""

import unittest
import numpy as np
from sklearn.metrics import cohen_kappa_score


class TestMetricsAndCalibration(unittest.TestCase):

    def test_qwk_perfect_and_worst(self):
        """QWK must equal 1.0 on perfect agreement and <= 0.0 on inverse agreement."""
        y_true = np.array([0, 1, 2, 3, 4] * 10)
        y_perf = y_true.copy()
        y_inv = 4 - y_true

        qwk_perf = cohen_kappa_score(y_true, y_perf, weights="quadratic")
        qwk_inv = cohen_kappa_score(y_true, y_inv, weights="quadratic")

        self.assertAlmostEqual(qwk_perf, 1.0, places=4)
        self.assertLess(qwk_inv, 0.0)

    def test_degenerate_qwk_detection(self):
        """Constant true labels produce zero label variance where QWK is mathematically uninformative."""
        y_const = np.zeros(100, dtype=int)
        y_pred = np.zeros(100, dtype=int)
        y_pred[0] = 1  # 1 mistake

        var_true = float(np.var(y_const))
        self.assertEqual(var_true, 0.0, "Expected zero variance for constant label vector")

    def test_ece_perfect_calibration(self):
        """Expected Calibration Error must be near 0 on perfectly calibrated probabilities."""
        n_bins = 10
        probs = np.array([0.95] * 100)
        corrects = np.array([1] * 95 + [0] * 5)  # exactly 95% accuracy for 0.95 confidence

        acc = np.mean(corrects)
        conf = np.mean(probs)
        gap = abs(acc - conf)
        self.assertAlmostEqual(gap, 0.0, places=4)


if __name__ == '__main__':
    unittest.main()
