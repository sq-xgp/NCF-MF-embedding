import unittest

import torch

from benchmark_ranking_losses import MF
from benchmark_side_features import FeatureMF
from train import set_seed


class TestFeatureMF(unittest.TestCase):
    def setUp(self):
        self.user_features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        self.item_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.users = torch.tensor([0, 1])
        self.items = torch.tensor([0, 1])

    def test_zero_initialized_side_projection_starts_at_exact_mf_control(self):
        set_seed(42)
        control = MF(2, 3, 4)
        for feature_set in ("user", "item", "both"):
            set_seed(42)
            variant = FeatureMF(
                2, 3, 4, self.user_features, self.item_features, feature_set
            )
            torch.testing.assert_close(
                control(self.users, self.items), variant(self.users, self.items)
            )

    def test_active_side_projection_receives_gradient(self):
        for feature_set in ("user", "item", "both"):
            set_seed(42)
            model = FeatureMF(
                2, 3, 4, self.user_features, self.item_features, feature_set
            )
            model(self.users, self.items).sum().backward()
            if model.user_projection is not None:
                self.assertGreater(model.user_projection.weight.grad.abs().sum().item(), 0)
            if model.item_projection is not None:
                self.assertGreater(model.item_projection.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
