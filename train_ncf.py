from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# Required by PyTorch for deterministic CUDA matrix multiplications.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from train import (
    MatrixFactorization,
    evaluate,
    load_ratings,
    make_loader,
    remove_cold_items,
    set_seed,
    temporal_split,
)


@dataclass(frozen=True)
class Config:
    data_path: str = "data/ml-100k/u.data"
    output_dir: str = "outputs/ncf"
    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.0
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 50
    patience: int = 5
    seed: int = 42
    evaluate_test: bool = True


class NeuralCollaborativeFiltering(nn.Module):
    """Single-tower NCF: concatenate ID embeddings, then score with an MLP."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        layers: list[nn.Module] = []
        input_dim = 2 * embedding_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(input_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.ReLU()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        output = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embedding(user_ids)
        item_vectors = self.item_embedding(item_ids)
        features = torch.cat((user_vectors, item_vectors), dim=-1)
        logits = self.mlp(features).squeeze(-1)
        return torch.sigmoid(logits)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run(config: Config) -> dict[str, object]:
    set_seed(config.seed)
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / config.data_path
    output_dir = project_dir / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(data_path)
    train, validation_all, test_all = temporal_split(ratings)
    validation, validation_cold_count = remove_cold_items(train, validation_all)
    test, test_cold_count = remove_cold_items(train, test_all)
    train_loader = make_loader(train, config.batch_size, True, config.seed)
    validation_loader = make_loader(validation, config.batch_size, False, config.seed)

    num_users = int(ratings["user_idx"].max()) + 1
    num_items = int(ratings["item_idx"].max()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralCollaborativeFiltering(
        num_users,
        num_items,
        config.embedding_dim,
        config.hidden_dims,
        config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    checkpoint_path = output_dir / "best_model.pt"

    best_validation_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_squared_error_sum = 0.0
        train_count = 0
        for users, items, targets in train_loader:
            users, items, targets = users.to(device), items.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_squared_error_sum += loss.item() * targets.numel()
            train_count += targets.numel()

        train_rating_rmse = 4.0 * math.sqrt(train_squared_error_sum / train_count)
        validation_metrics = evaluate(model, validation_loader, device)
        validation_rmse = validation_metrics["rating_rmse"]
        history.append(
            {
                "epoch": epoch,
                "train_rating_rmse": train_rating_rmse,
                "validation_rating_rmse": validation_rmse,
            }
        )
        print(
            f"epoch={epoch:02d} train_rmse={train_rating_rmse:.4f} "
            f"validation_rmse={validation_rmse:.4f}"
        )

        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"early stopping after epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_metrics = None
    if config.evaluate_test:
        test_loader = make_loader(test, config.batch_size, False, config.seed)
        test_metrics = evaluate(model, test_loader, device)

    mf_v2 = MatrixFactorization(num_users, num_items, config.embedding_dim, use_bias=True)
    results: dict[str, object] = {
        "model": "single_tower_ncf",
        "config": asdict(config),
        "device": str(device),
        "tensor_shapes": {
            "ids": "[batch]",
            "each_embedding": f"[batch, {config.embedding_dim}]",
            "concatenated": f"[batch, {2 * config.embedding_dim}]",
            "output": "[batch]",
        },
        "parameter_count": parameter_count(model),
        "mf_v2_parameter_count_same_embedding_dim": parameter_count(mf_v2),
        "mf_v2_reference": {
            "best_epoch": 2,
            "test_rating_rmse": 0.9874,
            "test_rating_mae": 0.7883,
        },
        "data": {
            "total_count": len(ratings),
            "train_count": len(train),
            "validation_total_count": len(validation_all),
            "validation_evaluated_count": len(validation),
            "validation_cold_item_count": validation_cold_count,
            "test_total_count": len(test_all),
            "test_evaluated_count": len(test),
            "test_cold_item_count": test_cold_count,
            "num_users": num_users,
            "num_items": num_items,
        },
        "best_epoch": best_epoch,
        "best_validation_rating_rmse": best_validation_rmse,
        "test": test_metrics,
        "history": history,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, indent=2))
    print(f"saved checkpoint to {checkpoint_path}")
    print(f"saved metrics to {metrics_path}")
    return results


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Minimal single-tower NCF for MovieLens 100K")
    parser.add_argument("--data-path", default=Config.data_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--embedding-dim", type=int, default=Config.embedding_dim)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=list(Config.hidden_dims))
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--validation-only",
        dest="evaluate_test",
        action="store_false",
        help="Tune on validation data without computing test metrics",
    )
    values = vars(parser.parse_args())
    values["hidden_dims"] = tuple(values["hidden_dims"])
    return Config(**values)


if __name__ == "__main__":
    run(parse_args())
