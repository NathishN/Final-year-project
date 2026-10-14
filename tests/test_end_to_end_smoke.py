"""
tests/test_end_to_end_smoke.py
==============================
End-to-end smoke test for CASMv2Model forward, backward, loss, and inference.
"""

import unittest
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, r"C:\fruit_dataset")
from phase50_casm_v2_model import CASMv2Model, compute_coral_loss, compute_effective_num_weights


class TestEndToEndSmoke(unittest.TestCase):

    def test_forward_backward_pass(self):
        """Forward pass, loss computation, and backprop must complete with finite gradients."""
        model = CASMv2Model(vis_dim=768, use_species=True, use_ood=True, num_classes=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        B = 16
        vis = torch.randn(B, 768)
        species = torch.randint(0, 6, (B,))
        ood = torch.randn(B, 2)
        targets = torch.randint(0, 5, (B,))

        cw = compute_effective_num_weights(targets, beta=0.999, num_classes=5)

        # Forward
        logits = model(vis, species, ood)
        self.assertEqual(logits.shape, (B, 4))

        # Loss
        loss = compute_coral_loss(logits, targets, class_weights=cw, num_classes=5)
        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any().item(), f"NaN grad in {name}")
                self.assertFalse(torch.isinf(param.grad).any().item(), f"Inf grad in {name}")

        # Inference
        ranks = model.predict_rank(logits)
        probs = model.predict_class_probs(logits)
        self.assertEqual(ranks.shape, (B,))
        self.assertEqual(probs.shape, (B, 5))


if __name__ == '__main__':
    unittest.main()
