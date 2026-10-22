import unittest
from pathlib import Path

import torch

from baselines import build_model, load_experiment_data


class BaselinesTest(unittest.TestCase):
    def test_rating_models_return_normalized_scores(self):
        for name in ("mf", "ncf"):
            with self.subTest(model=name):
                model = build_model(name, "rating", 3, 4, embedding_dim=8)
                scores = model(torch.tensor([0, 1]), torch.tensor([1, 2]))
                self.assertEqual(scores.shape, (2,))
                self.assertTrue(torch.all((scores >= 0) & (scores <= 1)).item())

    def test_ranking_models_return_logits(self):
        for name in ("mf", "ncf"):
            with self.subTest(model=name):
                model = build_model(name, "ranking", 3, 4, embedding_dim=8)
                scores = model(torch.tensor([0, 1]), torch.tensor([1, 2]))
                self.assertEqual(scores.shape, (2,))
                self.assertTrue(torch.isfinite(scores).all().item())

    def test_existing_data_protocol(self):
        if not (Path(__file__).resolve().parents[1] / "data/ml-100k/u.data").exists():
            self.skipTest("MovieLens 100K is not distributed with the source code")
        data = load_experiment_data()
        self.assertEqual(len(data.ratings), 100_000)
        self.assertEqual(len(data.train) + len(data.validation) + len(data.test), 100_000)
        self.assertEqual(data.num_users, 943)
        self.assertEqual(data.num_items, 1682)
        self.assertEqual(len(data.rating_validation) + data.validation_cold_items, len(data.validation))
        self.assertEqual(len(data.rating_test) + data.test_cold_items, len(data.test))

    def test_invalid_options_fail_fast(self):
        with self.assertRaises(ValueError):
            build_model("mf", "ranking", 3, 4, use_bias=True)
        with self.assertRaises(ValueError):
            build_model("ncf", "rating", 3, 4, use_bias=True)


if __name__ == "__main__":
    unittest.main()
