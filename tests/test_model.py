from __future__ import annotations

import unittest

import torch

from src.models.hybrid import IsolatedSignClassifier, STGCNTransformerClassifier
from src.models.stgcn import NODE_COUNT, build_landmark_graph, landmarks_to_graph


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

    def test_graph_conversion_preserves_hand_presence(self) -> None:
        landmarks = torch.zeros(2, 12, 258)
        landmarks[:, :, 3] = 0.75
        landmarks[:, :, 132] = 0.4

        graph = landmarks_to_graph(landmarks)

        self.assertEqual(graph.shape, (2, 4, 12, NODE_COUNT))
        self.assertTrue(torch.all(graph[:, 3, :, 33] == 1))
        self.assertTrue(torch.all(graph[:, 3, :, 54] == 0))

    def test_graph_contains_joint_edges_and_self_connections(self) -> None:
        source, target, weights = build_landmark_graph()
        edges = set(zip(source.tolist(), target.tolist(), strict=True))

        self.assertIn((15, 33), edges)
        self.assertIn((33, 15), edges)
        self.assertIn((16, 54), edges)
        self.assertIn((54, 16), edges)
        self.assertTrue(all((node, node) in edges for node in range(NODE_COUNT)))
        self.assertTrue(torch.isfinite(weights).all())

    def test_stgcn_transformer_forward_and_backward(self) -> None:
        model = STGCNTransformerClassifier(
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
        self.assertIsNotNone(model.graph_encoder.blocks[0].edge_importance.grad)
        self.assertIsNotNone(model.classifier[-1].weight.grad)

    def test_stgcn_transformer_rejects_non_mediapipe_feature_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "258 input features"):
            STGCNTransformerClassifier(num_classes=3, input_dim=100)


if __name__ == "__main__":
    unittest.main()
