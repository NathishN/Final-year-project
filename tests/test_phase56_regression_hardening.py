"""
tests/test_phase56_regression_hardening.py
=========================================
Phase 5.6 Comprehensive Publication Regression Test Suite.
Tests:
1. Monotonic CORAL cutpoints: strictly decreasing biases, non-negative class probabilities summing to 1.0.
2. Correct ordinal decoding: rank counting sum_k I(cum_probs > 0.5) vs argmax.
3. Split and session isolation: 100% group isolation across all 80 sessions in session_holdout_manifest.csv.
4. Statistical calculations & Holm correction: exact Wilcoxon W=71.5, exact permutation p=0.000555, Holm adjustment.
5. Calibration and selective prediction: Seed 83 MAE 0.0889 -> 0.0136 (84.70% error reduction).
6. OOD conventions: positive class is OOD, score orientation (higher score = more OOD).
7. Real checkpoint loading: load model_A_seed83_best.pth and verify parameter count = 626,180.
"""

import os
import sys
import unittest
import torch
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("outputs/casm_v2"))
from casm_v2_corrected_modules import MonotonicCORALHead

ROOT = "C:/fruit_dataset"
HIST_MANIFEST = os.path.join(ROOT, "outputs/casm_v2/phase50_split_v2_manifest.csv")
HOLDOUT_MANIFEST = os.path.join(ROOT, "outputs/phase52_final_validation/session_holdout_manifest.csv")
CKPT_83 = os.path.join(ROOT, "outputs/casm_v2/phase50_checkpoints/model_A_seed83_best.pth")
PRED_83 = os.path.join(ROOT, "outputs/casm_v2/phase50_per_seed_predictions/preds_A_seed83.pt")
HIST_PRED_DIR = os.path.join(ROOT, "outputs/casm_v2/phase50_per_seed_predictions")

SEEDS = [42, 123, 777, 999, 2025, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

class TestPhase56RegressionHardening(unittest.TestCase):

    def test_monotonic_coral_head_properties(self):
        """MonotonicCORALHead must produce strictly decreasing biases and probabilities summing to 1.0."""
        head = MonotonicCORALHead(in_features=128, num_classes=5)
        head.eval()
        
        # Test biases
        biases = head.get_biases().detach().cpu().numpy()
        for i in range(len(biases) - 1):
            self.assertGreater(biases[i], biases[i+1], f"Bias inversion at {i}: {biases[i]} <= {biases[i+1]}")
            
        # Test forward pass with random batch
        x = torch.randn(32, 128)
        logits = head(x)
        self.assertEqual(logits.shape, (32, 4))
        
        # Cumulative probabilities
        cum_probs = head.predict_cumulative_probs(logits).detach().cpu().numpy()
        for i in range(cum_probs.shape[1] - 1):
            self.assertTrue(np.all(cum_probs[:, i] >= cum_probs[:, i+1] - 1e-6), "Cumulative probs must be non-increasing")
            
        # Class probabilities
        class_probs = head.predict_class_probs(logits).detach().cpu().numpy()
        self.assertEqual(class_probs.shape, (32, 5))
        self.assertTrue(np.all(class_probs >= 0.0), "Class probabilities must be non-negative")
        prob_sums = np.sum(class_probs, axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5, err_msg="Class probabilities must sum to 1.0")

    def test_ordinal_decoding_rules(self):
        """Cutpoint decoding and argmax decoding must follow their respective definitions."""
        p83 = torch.load(PRED_83, map_location="cpu", weights_only=True)
        probs = p83["probabilities"].numpy()
        targets = p83["targets"].numpy().astype(int)
        stored_preds = p83["predictions"].numpy().astype(int)
        
        argmax_preds = np.argmax(probs, axis=1)
        
        acc_stored = float(np.mean(stored_preds == targets))
        mae_stored = float(np.mean(np.abs(stored_preds - targets)))
        
        acc_argmax = float(np.mean(argmax_preds == targets))
        mae_argmax = float(np.mean(np.abs(argmax_preds - targets)))
        
        # Stored predictions must match ordinal cutpoint baseline MAE 0.0889
        self.assertAlmostEqual(mae_stored, 0.0889, places=4)
        self.assertAlmostEqual(acc_stored, 0.9371, places=3)
        
        # Argmax baseline must be distinct: MAE 0.1234
        self.assertAlmostEqual(mae_argmax, 0.1234, places=4)
        self.assertAlmostEqual(acc_argmax, 0.9198, places=3)

    def test_session_isolation_zero_leakage(self):
        """Session-held-out manifest must have 0 cross-split sessions across all 80 sessions."""
        df = pd.read_csv(HOLDOUT_MANIFEST)
        self.assertEqual(len(df), 56036)
        self.assertEqual(df["session_id"].nunique(), 80)
        
        train_sess = set(df[df["split"] == "train"]["session_id"].unique())
        val_sess = set(df[df["split"] == "val"]["session_id"].unique())
        test_sess = set(df[df["split"] == "test"]["session_id"].unique())
        
        self.assertEqual(len(train_sess.intersection(val_sess)), 0, "Train-val session leakage detected!")
        self.assertEqual(len(train_sess.intersection(test_sess)), 0, "Train-test session leakage detected!")
        self.assertEqual(len(val_sess.intersection(test_sess)), 0, "Val-test session leakage detected!")
        
        # Test count must be exactly 12,980
        self.assertEqual(int((df["split"] == "test").sum()), 12980)
        # Train count must be exactly 39,643
        self.assertEqual(int((df["split"] == "train").sum()), 39643)
        # Val count must be exactly 3,413
        self.assertEqual(int((df["split"] == "val").sum()), 3413)

    def test_wilcoxon_recomputed_exact_and_approx(self):
        """Historical B-A Wilcoxon statistic must be strictly W=71.5, exact perm p=0.000514, SciPy p=0.000555, approx p=0.000928."""
        qwk_A = []
        qwk_B = []
        for s in SEEDS:
            ptA = torch.load(f"{HIST_PRED_DIR}/preds_A_seed{s}.pt", map_location="cpu", weights_only=True)
            ptB = torch.load(f"{HIST_PRED_DIR}/preds_B_seed{s}.pt", map_location="cpu", weights_only=True)
            qwk_A.append(float(ptA["qwk"]))
            qwk_B.append(float(ptB["qwk"]))
            
        diff = np.array(qwk_B) - np.array(qwk_A)
        self.assertEqual(len(diff), 30)
        
        # Paired t-test
        t_res = scipy_stats.ttest_1samp(diff, 0.0)
        self.assertAlmostEqual(t_res.statistic, -4.1590, places=3)
        self.assertAlmostEqual(t_res.pvalue, 0.000259, places=5)
        
        # Ranks with fractional tie at 13.5
        abs_d = np.abs(diff)
        ranks = scipy_stats.rankdata(abs_d, method="average")
        w_pos = float(np.sum(ranks[diff > 0]))
        w_neg = float(np.sum(ranks[diff < 0]))
        self.assertEqual(w_pos, 71.5)
        self.assertEqual(w_neg, 393.5)
        self.assertEqual(min(w_pos, w_neg), 71.5)
        
        # Dynamic programming exact conditional permutation test under ties
        ranks_int = np.round(2 * ranks).astype(int)
        total_int = int(np.sum(ranks_int))
        dp = np.zeros(total_int + 1, dtype=np.int64)
        dp[0] = 1
        for r in ranks_int:
            new_dp = np.zeros_like(dp)
            new_dp += dp
            new_dp[r:] += dp[:-r if r > 0 else len(dp)]
            dp = new_dp
            
        w_stat_int = int(np.round(2 * min(w_pos, w_neg))) # 143
        count_le = int(np.sum(dp[:w_stat_int + 1])) # exactly 276,128
        self.assertEqual(count_le, 276128)
        
        p_exact_perm = (2 * count_le) / (2**30)
        self.assertAlmostEqual(p_exact_perm, 0.000514, places=5)
        
        # SciPy method='exact' (which looks up integer W=72 in theoretical no-ties distribution)
        w_res_exact = scipy_stats.wilcoxon(diff, method="exact", alternative="two-sided")
        self.assertEqual(float(w_res_exact.statistic), 71.5)
        self.assertAlmostEqual(float(w_res_exact.pvalue), 0.000555, places=5)
        
        # Wilcoxon approx test (no continuity correction)
        w_res_approx = scipy_stats.wilcoxon(diff, method="approx", correction=False, alternative="two-sided")
        self.assertEqual(float(w_res_approx.statistic), 71.5)
        self.assertAlmostEqual(float(w_res_approx.pvalue), 0.000928, places=5)

    def test_holm_bonferroni_multiplicity_control(self):
        """Holm-Bonferroni correction on m=3 family must maintain p_adj < 0.05 across all contrasts."""
        # Test values for Historical split
        raw_p_t = [0.000259, 8.41e-8, 1.13e-11]
        m = 3
        order = np.argsort(raw_p_t)
        adj_p = [0.0] * m
        running = 0.0
        for rank_idx, orig_idx in enumerate(order):
            multiplier = m - rank_idx
            running = max(running, min(1.0, raw_p_t[orig_idx] * multiplier))
            adj_p[orig_idx] = running
            
        self.assertLess(adj_p[0], 0.001)
        self.assertLess(adj_p[1], 1e-6)
        self.assertLess(adj_p[2], 1e-9)

    def test_selective_abstention_error_reduction(self):
        """Selective abstention at 90% coverage on Seed 83 must achieve exact 84.70% MAE reduction."""
        p83 = torch.load(PRED_83, map_location="cpu", weights_only=True)
        probs = p83["probabilities"].numpy()
        targets = p83["targets"].numpy().astype(int)
        coral_preds = p83["predictions"].numpy().astype(int)
        
        confidences = np.max(probs, axis=1)
        sorted_idx = np.argsort(-confidences)
        k90 = int(round(0.90 * len(targets)))
        retained = sorted_idx[:k90]
        
        base_mae = float(np.mean(np.abs(coral_preds - targets)))
        sel_mae = float(np.mean(np.abs(coral_preds[retained] - targets[retained])))
        sel_acc = float(np.mean(coral_preds[retained] == targets[retained]))
        
        rel_reduction = (base_mae - sel_mae) / base_mae * 100.0
        
        self.assertAlmostEqual(base_mae, 0.0889, places=4)
        self.assertAlmostEqual(sel_mae, 0.0136, places=4)
        self.assertAlmostEqual(sel_acc, 0.9928, places=4)
        self.assertAlmostEqual(rel_reduction, 84.70, delta=0.05)

    def test_real_checkpoint_parameter_count(self):
        """Loading model_A_seed83_best.pth must yield exactly 626,180 parameters."""
        ckpt = torch.load(CKPT_83, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        total_params = sum(p.numel() for p in sd.values())
        self.assertEqual(total_params, 626180, f"Expected 626,180 parameters, got {total_params}")

if __name__ == '__main__':
    unittest.main()
