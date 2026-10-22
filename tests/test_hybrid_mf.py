import unittest
from pathlib import Path

import torch

from train import load_ratings
from train_hybrid_mf import HybridMatrixFactorization, load_side_features


PROJECT = Path(__file__).resolve().parents[1]


class TestHybridMF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ratings = load_ratings(PROJECT / "data/ml-100k/u.data")
        cls.user_features, cls.item_features, _ = load_side_features(
            cls.ratings,
            PROJECT / "data/ml-100k/u.user",
            PROJECT / "data/ml-100k/u.item",
        )

    def test_side_feature_alignment_and_shapes(self):
        self.assertEqual(tuple(self.user_features.shape), (943, 24))
        self.assertEqual(tuple(self.item_features.shape), (1682, 20))
        self.assertTrue(torch.isfinite(self.user_features).all())
        self.assertTrue(torch.isfinite(self.item_features).all())

    def test_output_shape_range_and_gradients(self):
        model = HybridMatrixFactorization(
            943, 1682, 8, self.user_features, self.item_features, "both"
        )
        output = model(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
        self.assertEqual(tuple(output.shape), (3,))
        self.assertTrue(torch.all((output > 0) & (output < 1)))
        output.sum().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)

    def test_feature_change_affects_prediction(self):
        model = HybridMatrixFactorization(
            943, 1682, 8, self.user_features, self.item_features, "both"
        )
        with torch.no_grad():
            before = model(torch.tensor([0]), torch.tensor([0]))
            model.item_features[0].zero_()
            after = model(torch.tensor([0]), torch.tensor([0]))
        self.assertFalse(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
