"""
tests/test_model_properties.py
==============================
Automated property tests for MonotonicCORALHead and RobustSpeciesEmbedding.
"""

import unittest
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, r"C:\fruit_dataset")
from phase50_casm_v2_model import MonotonicCORALHead, RobustSpeciesEmbedding, CASMv2Model


class TestModelProperties(unittest.TestCase):

    def test_cutpoint_strictly_decreasing(self):
        """MonotonicCORAL biases must satisfy b_0 > b_1 > b_2 > b_3 across 20 random seeds."""
        for s in range(20):
            torch.manual_seed(s)
            head = MonotonicCORALHead(in_features=64, num_classes=5)
            b = head.get_biases()
            diffs = b[:-1] - b[1:]
            self.assertTrue((diffs > 0).all().item(), f"Cutpoint inversion at seed {s}")

    def test_class_probs_valid_distribution(self):
        """Discrete class probabilities must be non-negative and sum to 1.0 within 1e-5."""
        head = MonotonicCORALHead(in_features=128, num_classes=5)
        for s in range(10):
            torch.manual_seed(s)
            x = torch.randn(32, 128)
            logits = head(x)
            probs = head.predict_class_probs(logits)
            self.assertTrue((probs >= 0.0).all().item(), "Negative class probability detected!")
            sums = probs.sum(dim=1)
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Probs do not sum to 1.0!")

    def test_unknown_species_routing(self):
        """Negative and out-of-range IDs must map safely to index 5 without exception."""
        emb = RobustSpeciesEmbedding(num_known_species=5, embedding_dim=16)
        unk_vec = emb.embedding(torch.tensor([5]))

        invalid_ids = torch.tensor([-1, -100, 5, 6, 999])
        out = emb(invalid_ids)
        for i in range(len(invalid_ids)):
            self.assertTrue(torch.equal(out[i:i+1], unk_vec), f"Invalid ID {invalid_ids[i]} did not route to unknown embedding!")

    def test_extreme_logit_stability(self):
        """Extreme input values (+/-1000) must produce finite outputs without NaN or Inf."""
        head = MonotonicCORALHead(in_features=128, num_classes=5)
        x_huge = torch.full((4, 128), 1000.0)
        x_tiny = torch.full((4, 128), -1000.0)

        for x in [x_huge, x_tiny]:
            logits = head(x)
            probs = head.predict_class_probs(logits)
            self.assertFalse(torch.isnan(probs).any().item(), "NaN in extreme prob prediction!")
            self.assertFalse(torch.isinf(probs).any().item(), "Inf in extreme prob prediction!")


if __name__ == '__main__':
    unittest.main()
