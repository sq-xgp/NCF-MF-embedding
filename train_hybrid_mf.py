from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch
from torch import nn

from train import (
    evaluate,
    load_ratings,
    make_loader,
    remove_cold_items,
    set_seed,
    temporal_split,
)


GENRES = (
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)


@dataclass(frozen=True)
class Config:
    data_path: str = "data/ml-100k/u.data"
    user_path: str = "data/ml-100k/u.user"
    item_path: str = "data/ml-100k/u.item"
    output_dir: str = "outputs/hybrid_mf"
    feature_set: str = "both"
    embedding_dim: int = 32
    batch_size: int = 1024
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 5
    seed: int = 42
    evaluate_test: bool = True


def _standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)


def load_side_features(
    ratings: pd.DataFrame, user_path: Path, item_path: Path
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build dense side-feature matrices aligned to the shared rating ID mapping."""
    users = pd.read_csv(
        user_path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        encoding="latin-1",
    )
    item_columns = ["item_id", "title", "release_date", "video_date", "url", *GENRES]
    items = pd.read_csv(
        item_path,
        sep="|",
        names=item_columns,
        encoding="latin-1",
    )

    user_ids = sorted(ratings["user_id"].unique())
    item_ids = sorted(ratings["item_id"].unique())
    users = users.set_index("user_id").loc[user_ids]
    items = items.set_index("item_id").loc[item_ids]
    if len(users) != len(user_ids) or len(items) != len(item_ids):
        raise ValueError("Side metadata does not cover the shared ID mapping")

    age = _standardize(torch.tensor(users["age"].to_numpy(), dtype=torch.float32)).unsqueeze(1)
    gender = torch.tensor(
        pd.get_dummies(users["gender"]).reindex(columns=["F", "M"], fill_value=False).to_numpy(),
        dtype=torch.float32,
    )
    occupations = sorted(users["occupation"].unique())
    occupation = torch.tensor(
        pd.get_dummies(users["occupation"]).reindex(columns=occupations, fill_value=False).to_numpy(),
        dtype=torch.float32,
    )
    user_features = torch.cat((age, gender, occupation), dim=1)

    genres = torch.tensor(items[list(GENRES)].to_numpy(dtype="float32"))
    years = pd.to_datetime(items["release_date"], errors="coerce").dt.year
    years = years.fillna(years.median())
    year = _standardize(torch.tensor(years.to_numpy(), dtype=torch.float32)).unsqueeze(1)
    item_features = torch.cat((genres, year), dim=1)
    metadata = {
        "user_features": ["standardized_age", "gender_F", "gender_M", *[f"occupation_{x}" for x in occupations]],
        "item_features": [*[f"genre_{x}" for x in GENRES], "standardized_release_year"],
    }
    return user_features, item_features, metadata


class HybridMatrixFactorization(nn.Module):
    """Biased MF plus a learned user-metadata/item-content interaction."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        feature_set: str = "both",
    ) -> None:
        super().__init__()
        if feature_set not in {"user", "item", "both"}:
            raise ValueError("feature_set must be one of: user, item, both")
        self.feature_set = feature_set
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.register_buffer("user_features", user_features)
        self.register_buffer("item_features", item_features)

        self.user_feature_projection = (
            nn.Linear(user_features.shape[1], embedding_dim)
            if feature_set in {"user", "both"}
            else None
        )
        self.item_feature_projection = (
            nn.Linear(item_features.shape[1], embedding_dim)
            if feature_set in {"item", "both"}
            else None
        )

        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for projection in (self.user_feature_projection, self.item_feature_projection):
            if projection is not None:
                nn.init.normal_(projection.weight, mean=0.0, std=0.01)
                nn.init.zeros_(projection.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embedding(user_ids)
        item_vectors = self.item_embedding(item_ids)
        logits = (user_vectors * item_vectors).sum(dim=-1)
        logits = (
            logits
            + self.global_bias
            + self.user_bias(user_ids).squeeze(-1)
            + self.item_bias(item_ids).squeeze(-1)
        )

        user_content = None
        item_content = None
        if self.user_feature_projection is not None:
            user_content = self.user_feature_projection(self.user_features[user_ids])
            logits = logits + (user_content * item_vectors).sum(dim=-1)
        if self.item_feature_projection is not None:
            item_content = self.item_feature_projection(self.item_features[item_ids])
            logits = logits + (user_vectors * item_content).sum(dim=-1)
        if user_content is not None and item_content is not None:
            logits = logits + (user_content * item_content).sum(dim=-1)
        return torch.sigmoid(logits)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run(config: Config) -> dict[str, object]:
    set_seed(config.seed)
    project_dir = Path(__file__).resolve().parent
    ratings = load_ratings(project_dir / config.data_path)
    user_features, item_features, feature_metadata = load_side_features(
        ratings, project_dir / config.user_path, project_dir / config.item_path
    )
    train, validation_all, test_all = temporal_split(ratings)
    validation, validation_cold_count = remove_cold_items(train, validation_all)
    test, test_cold_count = remove_cold_items(train, test_all)
    train_loader = make_loader(train, config.batch_size, True, config.seed)
    validation_loader = make_loader(validation, config.batch_size, False, config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridMatrixFactorization(
        int(ratings["user_idx"].max()) + 1,
        int(ratings["item_idx"].max()) + 1,
        config.embedding_dim,
        user_features,
        item_features,
        config.feature_set,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    output_dir = project_dir / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    best_validation_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        squared_error_sum = 0.0
        count = 0
        for users, items, targets in train_loader:
            users, items, targets = users.to(device), items.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            squared_error_sum += loss.item() * targets.numel()
            count += targets.numel()

        train_rmse = 4.0 * math.sqrt(squared_error_sum / count)
        validation_rmse = evaluate(model, validation_loader, device)["rating_rmse"]
        history.append({"epoch": epoch, "train_rating_rmse": train_rmse, "validation_rating_rmse": validation_rmse})
        print(f"epoch={epoch:02d} train_rmse={train_rmse:.4f} validation_rmse={validation_rmse:.4f}")
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
        test_metrics = evaluate(model, make_loader(test, config.batch_size, False, config.seed), device)

    results: dict[str, object] = {
        "model": "hybrid_biased_mf",
        "config": asdict(config),
        "device": str(device),
        "feature_metadata": feature_metadata,
        "feature_shapes": {"user": list(user_features.shape), "item": list(item_features.shape)},
        "parameter_count": parameter_count(model),
        "mf_v2_reference": {"parameter_count": 86626, "best_epoch": 2, "test_rating_rmse": 0.9874, "test_rating_mae": 0.7883},
        "data": {
            "total_count": len(ratings), "train_count": len(train),
            "validation_total_count": len(validation_all), "validation_evaluated_count": len(validation),
            "validation_cold_item_count": validation_cold_count,
            "test_total_count": len(test_all), "test_evaluated_count": len(test),
            "test_cold_item_count": test_cold_count,
        },
        "best_epoch": best_epoch,
        "best_validation_rating_rmse": best_validation_rmse,
        "test": test_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_rating_rmse": best_validation_rmse, "test": test_metrics}, indent=2))
    return results


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Feature-aware biased MF on MovieLens 100K")
    parser.add_argument("--data-path", default=Config.data_path)
    parser.add_argument("--user-path", default=Config.user_path)
    parser.add_argument("--item-path", default=Config.item_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--feature-set", choices=("user", "item", "both"), default=Config.feature_set)
    parser.add_argument("--embedding-dim", type=int, default=Config.embedding_dim)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--validation-only", dest="evaluate_test", action="store_false")
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
