import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("outputs/casm_v2"))

import unittest
import torch
import torch.nn as nn
from context_audit_train import CORALHead as OriginalCORALHead
from casm_v2_corrected_modules import MonotonicCORALHead, RobustSpeciesEmbedding


class TestCORALImplementationAudit(unittest.TestCase):
    
    def test_original_coral_failure_mode(self):
        """
        Demonstrates that unconstrained biases in OriginalCORALHead can invert,
        producing negative class probabilities.
        """
        orig_head = OriginalCORALHead(in_features=4, num_classes=5)
        # Manually set inverted biases: b_0 < b_1
        with torch.no_grad():
            orig_head.bias.copy_(torch.tensor([-1.0, 1.5, 0.5, -0.5]))
            orig_head.weight.zero_()
            
        x = torch.zeros((1, 4))
        logits = orig_head(x)  # [-1.0, 1.5, 0.5, -0.5]
        cum_probs = torch.sigmoid(logits)
        
        # P(Y > 0) = sigmoid(-1.0) = 0.2689
        # P(Y > 1) = sigmoid(1.5)  = 0.8176
        # P(Y = 1) = P(Y > 0) - P(Y > 1) = 0.2689 - 0.8176 = -0.5487 < 0 !
        p_class_1 = cum_probs[0, 0] - cum_probs[0, 1]
        self.assertTrue(p_class_1.item() < 0, "Original CORAL failed to reproduce negative probability on inverted bias")
        print(f"  [CONFIRMED] Original CORAL unconstrained bias allows negative class probability: P(Y=1) = {p_class_1.item():.4f}")

    def test_monotonic_coral_head_properties(self):
        """
        Verifies that MonotonicCORALHead guarantees monotonicity, valid probabilities,
        and sums to 1.0 under all inputs.
        """
        head = MonotonicCORALHead(in_features=128, num_classes=5)
        
        # 1. Output dimensions
        x = torch.randn(16, 128)
        logits = head(x)
        self.assertEqual(logits.shape, (16, 4))
        
        # 2. Bias monotonicity: b_0 > b_1 > b_2 > b_3
        biases = head.get_biases()
        diffs = biases[:-1] - biases[1:]
        self.assertTrue((diffs > 0).all().item(), "Biases are not strictly decreasing!")
        
        # 3. Cumulative probability monotonicity
        cum_probs = head.predict_cumulative_probs(logits)
        cum_diffs = cum_probs[:, :-1] - cum_probs[:, 1:]
        self.assertTrue((cum_diffs >= 0).all().item(), "Cumulative probabilities are not non-increasing!")
        
        # 4. Class probabilities: non-negative and sum to 1.0
        class_probs = head.predict_class_probs(logits)
        self.assertEqual(class_probs.shape, (16, 5))
        self.assertTrue((class_probs >= 0.0).all().item(), "Class probabilities contain negative values!")
        sums = class_probs.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Class probabilities do not sum to 1.0!")
        
        # 5. Rank prediction valid range [0..4]
        ranks = head.predict_rank(logits)
        self.assertTrue(((ranks >= 0) & (ranks <= 4)).all().item())

    def test_monotonic_coral_extreme_values_and_dtypes(self):
        """Tests numerical stability on extreme inputs and mixed precision."""
        head = MonotonicCORALHead(in_features=32, num_classes=5)
        
        # Extreme inputs
        x_extreme = torch.tensor([[-1000.0] * 32, [1000.0] * 32, [0.0] * 32])
        logits = head(x_extreme)
        class_probs = head.predict_class_probs(logits)
        self.assertFalse(torch.isnan(class_probs).any())
        self.assertTrue((class_probs >= 0.0).all().item())
        self.assertTrue(torch.allclose(class_probs.sum(dim=1), torch.ones(3), atol=1e-5))
        
        # Half precision
        head_half = head.half()
        x_half = torch.randn(8, 32, dtype=torch.float16)
        logits_half = head_half(x_half)
        probs_half = head_half.predict_class_probs(logits_half)
        self.assertFalse(torch.isnan(probs_half).any())
        self.assertTrue((probs_half >= 0.0).all().item())


class TestRobustSpeciesEmbedding(unittest.TestCase):
    
    def test_known_and_unknown_species_routing(self):
        """Verifies mapping of known IDs and fallback for unknown/invalid IDs."""
        module = RobustSpeciesEmbedding(embedding_dim=16)
        
        # Known IDs 0..4
        known_ids = torch.tensor([0, 1, 2, 3, 4])
        sanitized_known = module.sanitize_ids(known_ids)
        self.assertTrue(torch.equal(sanitized_known, known_ids))
        
        # Reserved unknown ID 5
        unk_id = torch.tensor([5])
        self.assertEqual(module.sanitize_ids(unk_id).item(), 5)
        
        # Negative IDs
        neg_ids = torch.tensor([-1, -10, -999])
        self.assertTrue(torch.equal(module.sanitize_ids(neg_ids), torch.tensor([5, 5, 5])))
        
        # Out-of-bounds IDs
        oob_ids = torch.tensor([6, 10, 1000])
        self.assertTrue(torch.equal(module.sanitize_ids(oob_ids), torch.tensor([5, 5, 5])))
        
        # Batched forward pass with mixed inputs
        mixed_batch = torch.tensor([0, -1, 3, 99, 4, 5, -5])
        emb = module(mixed_batch)
        self.assertEqual(emb.shape, (7, 16))
        
        # All unknown elements must yield identical embedding to index 5
        emb_ref_unk = module(torch.tensor([5]))[0]
        self.assertTrue(torch.equal(emb[1], emb_ref_unk))
        self.assertTrue(torch.equal(emb[3], emb_ref_unk))
        self.assertTrue(torch.equal(emb[5], emb_ref_unk))
        self.assertTrue(torch.equal(emb[6], emb_ref_unk))
        print("  [CONFIRMED] RobustSpeciesEmbedding successfully routed all invalid/unknown IDs to index 5.")


if __name__ == "__main__":
    unittest.main()
