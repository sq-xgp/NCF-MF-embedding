"""Ten controlled MF/NCF ranking experiments on the existing MovieLens split.

Explicit: all observed ratings 4-5 and 1-2, with binary targets.
Sampled: one item per positive and epoch from the union of train-unobserved
items and that user's observed train negatives (ratings 1-2).
Rating 3 is neutral and has no training label. Validation NDCG@K selects the
checkpoint; the test split is evaluated only after that selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from train import load_ratings, set_seed, temporal_split


EXPERIMENTS = (
    ("explicit", "mse"),
    ("explicit", "bce"),
    ("sampled", "mse"),
    ("sampled", "bce"),
    ("sampled", "bpr"),
)
SAMPLING_PROTOCOL = "train_unobserved_union_real_negative_v2"


@dataclass(frozen=True)
class Config:
    dim: int = 64
    batch_size: int = 1024
    lr: float = 0.001
    weight_decay: float = 1e-6
    epochs: int = 60
    patience: int = 5
    seed: int = 42
    k: int = 10
    score_batch_size: int = 65536
    output_dir: str = "outputs/ranking_benchmark_mixed_pool_100"


class MF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (
            (self.user_embedding(users) * self.item_embedding(items)).sum(-1)
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
            + self.global_bias
        )


class NCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(),
            nn.Linear(dim, dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.user_embedding(users), self.item_embedding(items)), -1)
        return self.mlp(features).squeeze(-1)


@dataclass
class RankingData:
    users: np.ndarray
    items: np.ndarray
    slices: list[tuple[int, int, set[int]]]
    eligible_users: int
    skipped_users: int
    cold_positives: int
    total_candidates: int


@dataclass
class SamplingData:
    users: list[int]
    positives: list[np.ndarray]
    pools: list[np.ndarray]


def build_ranking_data(
    target: pd.DataFrame, history: pd.DataFrame, known_items: set[int]
) -> RankingData:
    seen_by_user = {
        int(user): set(rows["item_idx"].astype(int))
        for user, rows in history.groupby("user_idx", sort=False)
    }
    flat_users: list[int] = []
    flat_items: list[int] = []
    slices: list[tuple[int, int, set[int]]] = []
    skipped = 0
    cold_positives = 0
    for user, rows in target.groupby("user_idx", sort=True):
        uid = int(user)
        positive = rows["rating"] > 3
        warm = rows["item_idx"].isin(known_items)
        relevant = set(rows.loc[positive & warm, "item_idx"].astype(int))
        cold_positives += int((positive & ~warm).sum())
        if not relevant:
            skipped += 1
            continue
        candidates = sorted(known_items - seen_by_user.get(uid, set()))
        if not relevant.issubset(candidates):
            raise AssertionError("A relevant item is missing from the candidate pool")
        start = len(flat_items)
        flat_items.extend(candidates)
        flat_users.extend([uid] * len(candidates))
        slices.append((start, len(flat_items), relevant))
    return RankingData(
        np.asarray(flat_users, dtype=np.int64),
        np.asarray(flat_items, dtype=np.int64),
        slices, len(slices), skipped, cold_positives, len(flat_items),
    )


@torch.inference_mode()
def evaluate_ranking(
    model: nn.Module, data: RankingData, k: int, device: torch.device,
    score_batch_size: int,
) -> dict[str, float | int]:
    model.eval()
    scores = np.empty(len(data.items), dtype=np.float32)
    for start in range(0, len(scores), score_batch_size):
        end = min(start + score_batch_size, len(scores))
        users = torch.as_tensor(data.users[start:end], device=device)
        items = torch.as_tensor(data.items[start:end], device=device)
        scores[start:end] = model(users, items).float().cpu().numpy()
    recalls: list[float] = []
    hits: list[float] = []
    ndcgs: list[float] = []
    for start, end, relevant in data.slices:
        local = scores[start:end]
        n = min(k, len(local))
        # Stable tie-breaking by item ID makes equal-score ranks reproducible.
        ranked = np.lexsort((data.items[start:end], -local))[:n]
        chosen = data.items[start:end][ranked]
        indicators = [int(int(item) in relevant) for item in chosen]
        hit_count = sum(indicators)
        recalls.append(hit_count / len(relevant))
        hits.append(float(hit_count > 0))
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(indicators))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
        ndcgs.append(dcg / ideal)
    return {
        "evaluated_users": data.eligible_users,
        "skipped_users_without_warm_positive": data.skipped_users,
        "excluded_cold_positive_count": data.cold_positives,
        "total_candidate_scores": data.total_candidates,
        f"recall@{k}": float(np.mean(recalls)),
        f"hit_ratio@{k}": float(np.mean(hits)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
    }


def observed_binary(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    rows = train.loc[train["rating"] != 3]
    users = rows["user_idx"].to_numpy(dtype=np.int64, copy=True)
    items = rows["item_idx"].to_numpy(dtype=np.int64, copy=True)
    labels = (rows["rating"].to_numpy() > 3).astype(np.float32, copy=True)
    return users, items, labels, {
        "positive": int(labels.sum()),
        "negative": int((1 - labels).sum()),
        "neutral_excluded": int((train["rating"] == 3).sum()),
    }


def prepare_sampling(
    train: pd.DataFrame, known_items: set[int], *, include_real_negatives: bool = True,
) -> SamplingData:
    item_pool = np.asarray(sorted(known_items), dtype=np.int64)
    users: list[int] = []
    positives: list[np.ndarray] = []
    pools: list[np.ndarray] = []
    for uid, rows in train.groupby("user_idx", sort=True):
        pos = rows.loc[rows["rating"] > 3, "item_idx"].to_numpy(dtype=np.int64)
        if not len(pos):
            continue
        seen = rows["item_idx"].to_numpy(dtype=np.int64)
        unobserved = item_pool[~np.isin(item_pool, seen)]
        if include_real_negatives:
            real_negatives = rows.loc[rows["rating"] < 3, "item_idx"].to_numpy(dtype=np.int64)
            pool = np.union1d(unobserved, real_negatives)
        else:
            pool = unobserved
        if not len(pool):
            raise ValueError(f"No eligible train negatives for user {uid}")
        users.append(int(uid))
        positives.append(pos)
        pools.append(pool)
    return SamplingData(users, positives, pools)


def sampled_pairs_from_prepared(
    prepared: SamplingData, epoch: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample shared epoch-specific pairs without inspecting future interactions."""
    rng = np.random.default_rng(seed + epoch)
    users: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    for uid, pos, pool in zip(prepared.users, prepared.positives, prepared.pools):
        sampled = rng.choice(pool, size=len(pos), replace=True)
        users.extend([uid] * len(pos))
        positives.extend(pos.tolist())
        negatives.extend(sampled.tolist())
    return (
        np.asarray(users, dtype=np.int64),
        np.asarray(positives, dtype=np.int64),
        np.asarray(negatives, dtype=np.int64),
    )


def sampled_pairs(
    train: pd.DataFrame, known_items: set[int], epoch: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw from train-unobserved items plus that user's real train negatives."""
    return sampled_pairs_from_prepared(prepare_sampling(train, known_items), epoch, seed)


def train_epoch(
    model: nn.Module, optimizer: torch.optim.Optimizer, regime: str, loss_name: str,
    train: pd.DataFrame, known_items: set[int], explicit: tuple[np.ndarray, np.ndarray, np.ndarray],
    sampling_data: SamplingData,
    epoch: int, config: Config, device: torch.device,
) -> tuple[float, int]:
    if regime == "explicit":
        users, items, labels = explicit
        dataset = TensorDataset(torch.from_numpy(users), torch.from_numpy(items), torch.from_numpy(labels))
    else:
        users, positives, negatives = sampled_pairs_from_prepared(sampling_data, epoch, config.seed)
        dataset = TensorDataset(torch.from_numpy(users), torch.from_numpy(positives), torch.from_numpy(negatives))
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(config.seed + epoch),
    )
    model.train()
    loss_total = 0.0
    count = 0
    for batch in loader:
        batch = tuple(value.to(device) for value in batch)
        optimizer.zero_grad()
        if regime == "explicit":
            user_ids, items, labels = batch
            logits = model(user_ids, items)
            loss = (
                F.mse_loss(torch.sigmoid(logits), labels)
                if loss_name == "mse" else F.binary_cross_entropy_with_logits(logits, labels)
            )
        else:
            user_ids, positives, negatives = batch
            positive_logits = model(user_ids, positives)
            negative_logits = model(user_ids, negatives)
            if loss_name == "bpr":
                loss = -F.logsigmoid(positive_logits - negative_logits).mean()
            else:
                logits = torch.cat((positive_logits, negative_logits))
                labels = torch.cat((torch.ones_like(positive_logits), torch.zeros_like(negative_logits)))
                loss = (
                    F.mse_loss(torch.sigmoid(logits), labels)
                    if loss_name == "mse" else F.binary_cross_entropy_with_logits(logits, labels)
                )
        loss.backward()
        optimizer.step()
        batch_count = len(batch[0])
        loss_total += loss.item() * batch_count
        count += batch_count
    return loss_total / count, count


def run_one(
    model_name: str, regime: str, loss_name: str, config: Config,
    train: pd.DataFrame, explicit: tuple[np.ndarray, np.ndarray, np.ndarray],
    sampling_data: SamplingData,
    known_items: set[int], validation_data: RankingData, test_data: RankingData,
    n_users: int, n_items: int, device: torch.device, output_root: Path,
    *, full_epochs: bool = False, resume: bool = False,
    checkpoint_every: int = 20, log_every: int = 20,
) -> dict[str, object]:
    if (regime, loss_name) not in EXPERIMENTS:
        raise ValueError("Unsupported regime/loss combination")
    set_seed(config.seed)
    model = (MF(n_users, n_items, config.dim) if model_name == "mf"
             else NCF(n_users, n_items, config.dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    run_dir = output_root / f"{model_name}_{regime}_{loss_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pt"
    progress_path = run_dir / "progress.pt"
    metrics_path = run_dir / "metrics.json"
    if regime == "sampled":
        if metrics_path.exists():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            if previous.get("sampling_protocol") != SAMPLING_PROTOCOL:
                raise ValueError(f"Existing results use another sampling protocol: {metrics_path}")
        if resume and progress_path.exists():
            previous_progress = torch.load(progress_path, map_location="cpu", weights_only=False)
            if previous_progress.get("sampling_protocol") != SAMPLING_PROTOCOL:
                raise ValueError(f"Existing progress uses another sampling protocol: {progress_path}")
    if resume and metrics_path.exists():
        completed = json.loads(metrics_path.read_text(encoding="utf-8"))
        if len(completed.get("history", [])) == config.epochs:
            print(f"{model_name}/{regime}/{loss_name}: already complete", flush=True)
            return completed
    best_ndcg = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if resume and progress_path.exists():
        saved = torch.load(progress_path, map_location=device, weights_only=False)
        if saved["config"] != asdict(config) or saved["experiment"] != (model_name, regime, loss_name):
            raise ValueError("Progress checkpoint does not match this experiment")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        best_ndcg = saved["best_ndcg"]
        best_epoch = saved["best_epoch"]
        stale = saved["stale"]
        history = saved["history"]
        start_epoch = saved["epoch"] + 1
        print(f"{model_name}/{regime}/{loss_name}: resumed at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, config.epochs + 1):
        train_loss, n_training_units = train_epoch(
            model, optimizer, regime, loss_name, train, known_items, explicit,
            sampling_data,
            epoch, config, device,
        )
        validation = evaluate_ranking(
            model, validation_data, config.k, device, config.score_batch_size,
        )
        ndcg = float(validation[f"ndcg@{config.k}"])
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "training_units": n_training_units,
            f"validation_recall@{config.k}": validation[f"recall@{config.k}"],
            f"validation_hit_ratio@{config.k}": validation[f"hit_ratio@{config.k}"],
            f"validation_ndcg@{config.k}": ndcg,
        })
        if ndcg > best_ndcg:
            best_ndcg, best_epoch, stale = ndcg, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if not full_epochs and stale >= config.patience:
                break
        if epoch == 1 or epoch % log_every == 0 or epoch == config.epochs:
            print(f"{model_name}/{regime}/{loss_name} epoch={epoch:04d} "
                  f"loss={train_loss:.4f} val_ndcg={ndcg:.4f} "
                  f"best_epoch={best_epoch}", flush=True)
        if epoch % checkpoint_every == 0 or epoch == config.epochs:
            temporary = progress_path.with_suffix(".tmp")
            torch.save({
                "config": asdict(config),
                "experiment": (model_name, regime, loss_name),
                "sampling_protocol": SAMPLING_PROTOCOL if regime == "sampled" else None,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_ndcg": best_ndcg,
                "best_epoch": best_epoch,
                "stale": stale,
                "history": history,
            }, temporary)
            os.replace(temporary, progress_path)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test = evaluate_ranking(model, test_data, config.k, device, config.score_batch_size)
    result: dict[str, object] = {
        "model": model_name, "regime": regime, "loss": loss_name,
        "sampling_protocol": SAMPLING_PROTOCOL if regime == "sampled" else None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_ndcg": best_ndcg,
        "best_validation": history[best_epoch - 1],
        "test": test,
        "history": history,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mf", "ncf"))
    parser.add_argument("--regime", choices=("explicit", "sampled"))
    parser.add_argument("--loss", choices=("mse", "bce", "bpr"))
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--full-epochs", action="store_true", help="Train every epoch without early stopping")
    parser.add_argument("--resume", action="store_true", help="Resume matching progress checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.checkpoint_every, args.log_every) <= 0:
        raise ValueError("epochs, patience, checkpoint-every, and log-every must be positive")
    config = Config(epochs=args.epochs, patience=args.patience, output_dir=args.output_dir)
    project_dir = Path(__file__).resolve().parent
    output_root = project_dir / config.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    ratings = load_ratings(project_dir / "data/ml-100k/u.data")
    train, validation, test = temporal_split(ratings)
    known_items = set(train["item_idx"].astype(int))
    observed_users, observed_items, observed_labels, counts = observed_binary(train)
    explicit = (observed_users, observed_items, observed_labels)
    sampling_data = prepare_sampling(train, known_items)
    validation_data = build_ranking_data(validation, train, known_items)
    test_history = pd.concat((train, validation), ignore_index=True)
    test_data = build_ranking_data(test, test_history, known_items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({
        "device": str(device), "split_counts": [len(train), len(validation), len(test)],
        "explicit_train_counts": counts,
        "validation_users": validation_data.eligible_users,
        "test_users": test_data.eligible_users,
    }), flush=True)
    selected = [(args.regime, args.loss)] if args.regime and args.loss else list(EXPERIMENTS)
    models = [args.model] if args.model else ["mf", "ncf"]
    results = []
    for model_name in models:
        for regime, loss_name in selected:
            results.append(run_one(
                model_name, regime, loss_name, config, train, explicit, sampling_data, known_items,
                validation_data, test_data,
                int(ratings["user_idx"].max()) + 1,
                int(ratings["item_idx"].max()) + 1,
                device, output_root,
                full_epochs=args.full_epochs,
                resume=args.resume,
                checkpoint_every=args.checkpoint_every,
                log_every=args.log_every,
            ))
    summary = {
        "config": asdict(config),
        "protocol": {
            "data": "existing MovieLens 100K u.data and temporal_split()",
            "positive": "rating > 3 (ratings 4 and 5)",
            "explicit_negative": "rating < 3 (ratings 1 and 2)",
            "neutral": "rating 3: no training target; remains an unknown test candidate",
            "explicit_training": "all observed positive and negative train records",
            "sampled_training": "one item from the union of user's observed train negatives (ratings 1-2) and train-unobserved globally train-known items per train positive; uniform with replacement, resampled each epoch; same pairs for every run",
            "sampled_negative_caveat": "unobserved items are unlabeled, not confirmed disliked; future interactions are not inspected while sampling",
            "candidate_items": "all globally train-known items excluding the user's earlier interactions",
            "validation_history": "train interactions",
            "test_history": "train + validation interactions",
            "relevance": "warm target items with rating > 3",
            "aggregation": "macro over users with at least one warm positive target item",
            "hit_ratio": "fraction of evaluated users with >=1 relevant item in Top-K",
            "checkpoint_selection": "highest validation nDCG@K, earliest epoch on ties",
            "test_usage": "once per run after checkpoint selection",
        },
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "explicit_train_counts": counts,
        "validation_population": {
            "eligible_users": validation_data.eligible_users,
            "skipped_users": validation_data.skipped_users,
            "cold_positives": validation_data.cold_positives,
        },
        "test_population": {
            "eligible_users": test_data.eligible_users,
            "skipped_users": test_data.skipped_users,
            "cold_positives": test_data.cold_positives,
        },
        "results": [{key: val for key, val in result.items() if key != "history"} for result in results],
    }
    if len(results) == 10:
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed": len(results), "output_dir": str(output_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
