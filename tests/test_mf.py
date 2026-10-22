import unittest

import pandas as pd
import torch

from recommend import recommend
from evaluate_ranking import recall_ndcg_at_k
from train import MatrixFactorization, RatingsDataset, remove_cold_items, temporal_split
from train_bpr import bpr_loss


class TestBaseline(unittest.TestCase):
    def test_bpr_prefers_correct_order(self):
        correct = bpr_loss(torch.tensor([3.0]), torch.tensor([1.0]))
        incorrect = bpr_loss(torch.tensor([1.0]), torch.tensor([3.0]))
        self.assertLess(correct.item(), incorrect.item())

    def test_recall_and_ndcg_reward_earlier_hits(self):
        relevant = {1, 2, 3}
        recall_late, ndcg_late = recall_ndcg_at_k([4, 1, 5, 2, 6], relevant, 5)
        recall_early, ndcg_early = recall_ndcg_at_k([1, 2, 4, 5, 6], relevant, 5)
        self.assertAlmostEqual(recall_late, 2 / 3)
        self.assertAlmostEqual(recall_early, 2 / 3)
        self.assertGreater(ndcg_early, ndcg_late)

    def test_rating_normalization(self):
        frame = pd.DataFrame(
            {"user_idx": [0, 0, 0], "item_idx": [0, 1, 2], "rating": [1, 3, 5]}
        )
        dataset = RatingsDataset(frame)
        self.assertTrue(torch.equal(dataset.targets, torch.tensor([0.0, 0.5, 1.0])))

    def test_model_output_shape_and_range(self):
        model = MatrixFactorization(3, 4, 2)
        output = model(torch.tensor([0, 1]), torch.tensor([2, 3]))
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.all((output > 0) & (output < 1)))

    def test_bias_terms_shift_prediction(self):
        model = MatrixFactorization(2, 2, 2, use_bias=True)
        with torch.no_grad():
            model.user_embedding.weight.zero_()
            model.item_embedding.weight.zero_()
            model.global_bias.zero_()
            model.user_bias.weight.zero_()
            model.item_bias.weight.zero_()
            model.user_bias.weight[1] = 1.0
        low = model(torch.tensor([0]), torch.tensor([0]))
        high = model(torch.tensor([1]), torch.tensor([0]))
        self.assertLess(low.item(), high.item())

    def test_temporal_split(self):
        frame = pd.DataFrame(
            {
                "user_id": [1] * 10,
                "item_id": list(range(10)),
                "rating": [3] * 10,
                "timestamp": list(range(10)),
                "user_idx": [0] * 10,
                "item_idx": list(range(10)),
            }
        )
        train, validation, test = temporal_split(frame)
        self.assertEqual(train["item_id"].tolist(), list(range(8)))
        self.assertEqual(validation["item_id"].tolist(), [8])
        self.assertEqual(test["item_id"].tolist(), [9])

    def test_cold_items_are_counted_and_removed(self):
        train = pd.DataFrame({"item_idx": [0, 1]})
        evaluation = pd.DataFrame({"item_idx": [1, 2, 0, 3]})
        warm, cold_count = remove_cold_items(train, evaluation)
        self.assertEqual(warm["item_idx"].tolist(), [1, 0])
        self.assertEqual(cold_count, 2)

    def test_recommend_only_returns_candidates(self):
        model = MatrixFactorization(1, 5, 2)
        with torch.no_grad():
            model.user_embedding.weight[0] = torch.tensor([1.0, 0.0])
            model.item_embedding.weight[:] = torch.tensor(
                [[0.0, 0.0], [3.0, 0.0], [1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]]
            )
        ranked = recommend(model, 0, [2, 3, 4], top_k=2, device=torch.device("cpu"))
        self.assertEqual([item for item, _ in ranked], [3, 2])


if __name__ == "__main__":
    unittest.main()
