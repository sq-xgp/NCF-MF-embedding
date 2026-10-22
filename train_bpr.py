from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_ranking import recall_ndcg_at_k
from train import load_ratings, set_seed, temporal_split


@dataclass(frozen=True)
class BPRConfig:
    data_path: str = "data/ml-100k/u.data"
    output_dir: str = "outputs/bpr"
    embedding_dim: int = 32
    batch_size: int = 1024
    learning_rate: float = 1e-2
    weight_decay: float = 1e-6
    epochs: int = 30
    patience: int = 5
    seed: int = 42
    k: int = 10
    positive_threshold: int = 4
    explicit_negative_max_rating: int = 2
    explicit_negative_probability: float = 0.5


class BPRModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        interaction = (
            self.user_embedding(users) * self.item_embedding(items)
        ).sum(dim=-1)
        return interaction + self.item_bias(items).squeeze(-1)

    def forward(
        self,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.score(users, positive_items), self.score(users, negative_items)


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(positive_scores - negative_scores).mean()


class TripletDataset(Dataset):
    def __init__(self, triplets: list[tuple[int, int, int]]) -> None:
        self.triplets = torch.tensor(triplets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.triplets[index]
        return row[0], row[1], row[2]


def sample_triplets(
    train: pd.DataFrame,
    positive_threshold: int,
    explicit_negative_max_rating: int,
    explicit_negative_probability: float,
    seed: int,
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    known_items = sorted(train["item_idx"].astype(int).unique().tolist())
    triplets: list[tuple[int, int, int]] = []

    for _, rows in train.groupby("user_idx", sort=True):
        user_idx = int(rows["user_idx"].iloc[0])
        positives = rows.loc[
            rows["rating"] >= positive_threshold, "item_idx"
        ].astype(int).tolist()
        explicit_negatives = rows.loc[
            rows["rating"] <= explicit_negative_max_rating, "item_idx"
        ].astype(int).tolist()
        seen = set(rows["item_idx"].astype(int).tolist())
        unobserved = [item for item in known_items if item not in seen]
        if not unobserved and not explicit_negatives:
            continue

        for positive_item in positives:
            use_explicit = explicit_negatives and (
                rng.random() < explicit_negative_probability or not unobserved
            )
            pool = explicit_negatives if use_explicit else unobserved
            negative_item = rng.choice(pool)
            triplets.append((user_idx, positive_item, negative_item))
    return triplets


@torch.no_grad()
def ranking_metrics(
    model: BPRModel,
    target: pd.DataFrame,
    history: pd.DataFrame,
    known_items: set[int],
    positive_threshold: int,
    k: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    history_by_user = {
        int(user): set(rows["item_idx"].astype(int).tolist())
        for user, rows in history.groupby("user_id", sort=False)
    }
    recalls: list[float] = []
    ndcgs: list[float] = []
    skipped = 0
    cold_positives = 0

    for user_id, rows in target.groupby("user_id", sort=True):
        user_idx = int(rows["user_idx"].iloc[0])
        positive_mask = rows["rating"] >= positive_threshold
        warm_mask = rows["item_idx"].isin(known_items)
        relevant = set(rows.loc[positive_mask & warm_mask, "item_idx"].astype(int))
        cold_positives += int((positive_mask & ~warm_mask).sum())
        if not relevant:
            skipped += 1
            continue
        candidates = sorted(known_items - history_by_user.get(int(user_id), set()))
        item_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        user_tensor = torch.full_like(item_tensor, user_idx)
        scores = model.score(user_tensor, item_tensor)
        top_positions = torch.topk(scores, k=min(k, len(candidates))).indices.cpu().tolist()
        recommended = [candidates[position] for position in top_positions]
        recall, ndcg = recall_ndcg_at_k(recommended, relevant, k)
        recalls.append(recall)
        ndcgs.append(ndcg)

    return {
        "evaluated_users": len(recalls),
        "skipped_users_without_warm_positive": skipped,
        "excluded_cold_positive_count": cold_positives,
        f"recall@{k}": sum(recalls) / len(recalls),
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
    }


def run(config: BPRConfig) -> dict[str, object]:
    set_seed(config.seed)
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ratings = load_ratings(project_dir / config.data_path)
    train, validation, test = temporal_split(ratings)
    known_items = set(train["item_idx"].astype(int).unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPRModel(
        int(ratings["user_idx"].max()) + 1,
        int(ratings["item_idx"].max()) + 1,
        config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint_path = output_dir / "best_model.pt"
    best_ndcg = -1.0
    best_epoch = 0
    stale_epochs = 0
    history_metrics: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        triplets = sample_triplets(
            train,
            config.positive_threshold,
            config.explicit_negative_max_rating,
            config.explicit_negative_probability,
            config.seed + epoch,
        )
        loader = DataLoader(
            TripletDataset(triplets),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + epoch),
            num_workers=0,
        )
        model.train()
        loss_sum = 0.0
        count = 0
        for users, positives, negatives in loader:
            users = users.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            optimizer.zero_grad()
            positive_scores, negative_scores = model(users, positives, negatives)
            loss = bpr_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * users.numel()
            count += users.numel()

        validation_metrics = ranking_metrics(
            model,
            validation,
            train,
            known_items,
            config.positive_threshold,
            config.k,
            device,
        )
        validation_ndcg = float(validation_metrics[f"ndcg@{config.k}"])
        epoch_result = {
            "epoch": epoch,
            "train_bpr_loss": loss_sum / count,
            f"validation_recall@{config.k}": validation_metrics[f"recall@{config.k}"],
            f"validation_ndcg@{config.k}": validation_ndcg,
        }
        history_metrics.append(epoch_result)
        print(json.dumps(epoch_result))

        if validation_ndcg > best_ndcg:
            best_ndcg = validation_ndcg
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"early stopping after epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_history = pd.concat([train, validation], ignore_index=True)
    test_metrics = ranking_metrics(
        model,
        test,
        test_history,
        known_items,
        config.positive_threshold,
        config.k,
        device,
    )
    result: dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "best_epoch": best_epoch,
        f"best_validation_ndcg@{config.k}": best_ndcg,
        "test": test_metrics,
        "history": history_metrics,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, indent=2))
    print(f"saved checkpoint to {checkpoint_path}")
    print(f"saved metrics to {metrics_path}")
    return result


def parse_args() -> BPRConfig:
    parser = argparse.ArgumentParser(description="MovieLens 100K BPR MF baseline")
    parser.add_argument("--data-path", default=BPRConfig.data_path)
    parser.add_argument("--output-dir", default=BPRConfig.output_dir)
    parser.add_argument("--embedding-dim", type=int, default=BPRConfig.embedding_dim)
    parser.add_argument("--batch-size", type=int, default=BPRConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=BPRConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=BPRConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=BPRConfig.epochs)
    parser.add_argument("--patience", type=int, default=BPRConfig.patience)
    parser.add_argument("--seed", type=int, default=BPRConfig.seed)
    parser.add_argument("--k", type=int, default=BPRConfig.k)
    parser.add_argument("--positive-threshold", type=int, default=BPRConfig.positive_threshold)
    parser.add_argument(
        "--explicit-negative-max-rating",
        type=int,
        default=BPRConfig.explicit_negative_max_rating,
    )
    parser.add_argument(
        "--explicit-negative-probability",
        type=float,
        default=BPRConfig.explicit_negative_probability,
    )
    return BPRConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
