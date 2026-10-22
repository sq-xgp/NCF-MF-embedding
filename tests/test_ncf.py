import unittest

import pandas as pd
import torch

from train import RatingsDataset, temporal_split
from train_ncf import NeuralCollaborativeFiltering


class TestNCF(unittest.TestCase):
    def test_output_shape_and_range(self):
        model = NeuralCollaborativeFiltering(3, 4, embedding_dim=2, hidden_dims=(4, 2))
        output = model(torch.tensor([0, 1]), torch.tensor([2, 3]))
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.all((output > 0) & (output < 1)))

    def test_all_parameters_receive_gradients(self):
        model = NeuralCollaborativeFiltering(3, 4, embedding_dim=2, hidden_dims=(4, 2))
        predictions = model(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
        torch.nn.functional.mse_loss(predictions, torch.tensor([0.0, 0.5, 1.0])).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_reuses_rating_labels_and_temporal_split(self):
        frame = pd.DataFrame(
            {
                "user_id": [1] * 10,
                "item_id": list(range(10)),
                "rating": [1, 3, 5, 1, 3, 5, 1, 3, 5, 1],
                "timestamp": list(range(10)),
                "user_idx": [0] * 10,
                "item_idx": list(range(10)),
            }
        )
        train, validation, test = temporal_split(frame)
        self.assertEqual((len(train), len(validation), len(test)), (8, 1, 1))
        dataset = RatingsDataset(frame.iloc[:3])
        self.assertTrue(torch.equal(dataset.targets, torch.tensor([0.0, 0.5, 1.0])))


if __name__ == "__main__":
    unittest.main()
