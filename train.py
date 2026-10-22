from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Config:
    data_path: str = "data/ml-100k/u.data"
    output_dir: str = "outputs"
    embedding_dim: int = 32
    batch_size: int = 1024
    learning_rate: float = 1e-2
    weight_decay: float = 1e-6
    epochs: int = 30
    patience: int = 5
    seed: int = 42
    use_bias: bool = False
    evaluate_test: bool = True


class RatingsDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.users = torch.tensor(frame["user_idx"].to_numpy(), dtype=torch.long)
        self.items = torch.tensor(frame["item_idx"].to_numpy(), dtype=torch.long)
        # Align the explicit 1--5 rating interval exactly to 0--1.
        self.targets = torch.tensor(
            ((frame["rating"].to_numpy(dtype="float32") - 1.0) / 4.0),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return self.users[index], self.items[index], self.targets[index]


class MatrixFactorization(nn.Module):
    """Minimal two-tower MF: ID embeddings, dot product, sigmoid."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.use_bias = use_bias
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        if use_bias:
            self.global_bias = nn.Parameter(torch.zeros(()))
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embedding(user_ids)
        item_vectors = self.item_embedding(item_ids)
        logits = (user_vectors * item_vectors).sum(dim=-1)
        if self.use_bias:
            logits = (
                logits
                + self.global_bias
                + self.user_bias(user_ids).squeeze(-1)
                + self.item_bias(item_ids).squeeze(-1)
            )
        return torch.sigmoid(logits)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_ratings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download MovieLens 100K and place u.data there."
        )
    frame = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": "int64", "item_id": "int64", "rating": "int64", "timestamp": "int64"},
    )
    if len(frame) != 100_000:
        raise ValueError(f"Expected MovieLens 100K (100000 rows), found {len(frame)} rows")
    if not frame["rating"].between(1, 5).all():
        raise ValueError("Ratings must be in the closed interval [1, 5]")

    # Stable, shared mappings across all versions and all splits.
    user_ids = sorted(frame["user_id"].unique())
    item_ids = sorted(frame["item_id"].unique())
    user_to_idx = {value: index for index, value in enumerate(user_ids)}
    item_to_idx = {value: index for index, value in enumerate(item_ids)}
    frame["user_idx"] = frame["user_id"].map(user_to_idx)
    frame["item_idx"] = frame["item_id"].map(item_to_idx)
    return frame


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user chronological 80/10/10 split with deterministic tie-breaking."""
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, records in frame.groupby("user_id", sort=True):
        records = records.sort_values(["timestamp", "item_id"], kind="stable")
        count = len(records)
        train_end = int(0.8 * count)
        validation_end = train_end + int(0.1 * count)
        train_parts.append(records.iloc[:train_end])
        validation_parts.append(records.iloc[train_end:validation_end])
        test_parts.append(records.iloc[validation_end:])

    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(validation_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    if len(train) + len(validation) + len(test) != len(frame):
        raise RuntimeError("Split lost or duplicated interactions")
    return train, validation, test


def remove_cold_items(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    known_items = set(train["item_idx"].unique())
    warm_mask = evaluation["item_idx"].isin(known_items)
    cold_count = int((~warm_mask).sum())
    return evaluation.loc[warm_mask].reset_index(drop=True), cold_count


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    count = 0
    for users, items, targets in loader:
        users, items, targets = users.to(device), items.to(device), targets.to(device)
        predictions = model(users, items)
        errors = predictions - targets
        squared_error_sum += errors.square().sum().item()
        absolute_error_sum += errors.abs().sum().item()
        count += targets.numel()
    if count == 0:
        raise ValueError("Evaluation split is empty after cold-item filtering")
    normalized_mse = squared_error_sum / count
    return {
        "normalized_mse": normalized_mse,
        "normalized_rmse": math.sqrt(normalized_mse),
        "rating_rmse": 4.0 * math.sqrt(normalized_mse),
        "rating_mae": 4.0 * absolute_error_sum / count,
        "count": count,
    }


def make_loader(
    frame: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        RatingsDataset(frame),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatrixFactorization(
        num_users=int(ratings["user_idx"].max()) + 1,
        num_items=int(ratings["item_idx"].max()) + 1,
        embedding_dim=config.embedding_dim,
        use_bias=config.use_bias,
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
    results: dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "data": {
            "total_count": len(ratings),
            "train_count": len(train),
            "validation_total_count": len(validation_all),
            "validation_evaluated_count": len(validation),
            "validation_cold_item_count": validation_cold_count,
            "test_total_count": len(test_all),
            "test_evaluated_count": len(test),
            "test_cold_item_count": test_cold_count,
            "num_users": ratings["user_idx"].nunique(),
            "num_items": ratings["item_idx"].nunique(),
        },
        "best_epoch": best_epoch,
        "best_validation_rating_rmse": best_validation_rmse,
        "test": test_metrics,
        "history": history,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"best_epoch": best_epoch, "test": test_metrics}
    if not config.evaluate_test:
        summary["note"] = "validation-only run; test metrics were intentionally not computed"
    print(json.dumps(summary, indent=2))
    print(f"saved checkpoint to {checkpoint_path}")
    print(f"saved metrics to {metrics_path}")
    return results


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Minimal MovieLens 100K MF two-tower baseline")
    parser.add_argument("--data-path", default=Config.data_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--embedding-dim", type=int, default=Config.embedding_dim)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--use-bias",
        action="store_true",
        help="Add global, user, and item bias terms before sigmoid",
    )
    parser.add_argument(
        "--validation-only",
        dest="evaluate_test",
        action="store_false",
        help="Tune on validation data without computing test metrics",
    )
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
