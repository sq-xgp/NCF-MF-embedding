import unittest

import numpy as np
import pandas as pd
import torch

from benchmark_ranking_losses import (
    build_ranking_data,
    evaluate_ranking,
    observed_binary,
    prepare_sampling,
    sampled_pairs,
)


class FixedScores(torch.nn.Module):
    def forward(self, users, items):
        return items.float()


class TestRankingBenchmark(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "user_idx": [0, 0, 0, 1, 1, 1],
            "item_idx": [0, 1, 2, 0, 3, 4],
            "rating": [5, 2, 3, 4, 1, 3],
        })

    def test_binary_labels_exclude_neutral_and_keep_all_real_negatives(self):
        users, items, labels, counts = observed_binary(self.train)
        self.assertEqual(counts, {"positive": 2, "negative": 2, "neutral_excluded": 2})
        self.assertEqual(set(zip(users.tolist(), items.tolist(), labels.tolist())), {
            (0, 0, 1.0), (0, 1, 0.0), (1, 0, 1.0), (1, 3, 0.0),
        })

    def test_sampled_pool_adds_real_train_negatives_but_not_neutral_or_positive(self):
        prepared = prepare_sampling(self.train, set(range(6)))
        self.assertEqual(prepared.users, [0, 1])
        np.testing.assert_array_equal(prepared.pools[0], np.array([1, 3, 4, 5]))
        np.testing.assert_array_equal(prepared.pools[1], np.array([1, 2, 3, 5]))

    def test_legacy_unobserved_only_pool_is_available_for_side_feature_control(self):
        prepared = prepare_sampling(self.train, set(range(6)), include_real_negatives=False)
        np.testing.assert_array_equal(prepared.pools[0], np.array([3, 4, 5]))
        np.testing.assert_array_equal(prepared.pools[1], np.array([1, 2, 5]))

    def test_sampled_negatives_come_only_from_union_pool(self):
        users, positives, negatives = sampled_pairs(self.train, set(range(6)), 1, 42)
        self.assertEqual(len(users), 2)
        pools = {0: {1, 3, 4, 5}, 1: {1, 2, 3, 5}}
        for user, positive, negative in zip(users, positives, negatives):
            self.assertEqual(int(positive), 0)
            self.assertIn(int(negative), pools[int(user)])

    def test_recall_hit_ratio_and_ndcg_use_same_candidates(self):
        history = pd.DataFrame({"user_idx": [0], "item_idx": [0], "rating": [5]})
        target = pd.DataFrame({
            "user_idx": [0, 0, 0],
            "item_idx": [1, 2, 3],
            "rating": [2, 3, 5],
        })
        data = build_ranking_data(target, history, {0, 1, 2, 3})
        result = evaluate_ranking(FixedScores(), data, 1, torch.device("cpu"), 4)
        self.assertEqual(result["recall@1"], 1.0)
        self.assertEqual(result["hit_ratio@1"], 1.0)
        self.assertEqual(result["ndcg@1"], 1.0)
        np.testing.assert_array_equal(data.items, np.array([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
