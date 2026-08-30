from __future__ import annotations

import unittest

import torch

from src.models.hybrid import IsolatedSignClassifier


class IsolatedSignClassifierTests(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        model = IsolatedSignClassifier(
            num_classes=3,
            sequence_length=12,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            feedforward_dim=64,
        )
        logits = model(torch.randn(2, 12, 258))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2]))
        loss.backward()

        self.assertEqual(logits.shape, (2, 3))
        self.assertIsNotNone(model.classifier[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
