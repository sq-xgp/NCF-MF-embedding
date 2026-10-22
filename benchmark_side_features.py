"""Ablate user and item metadata on the strongest MF+BPR ranking protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from benchmark_ranking_losses import (
    Config,
    MF,
    build_ranking_data,
    evaluate_ranking,
    observed_binary,
    prepare_sampling,
    train_epoch,
)
from train import load_ratings, set_seed, temporal_split
from train_hybrid_mf import load_side_features


FEATURE_SETS = ("none", "user", "item", "both")


class FeatureMF(MF):
    """Biased MF with additive metadata projections into the same latent space."""

    def __init__(
        self, n_users: int, n_items: int, dim: int,
        user_features: torch.Tensor, item_features: torch.Tensor, feature_set: str,
    ) -> None:
        if feature_set not in FEATURE_SETS[1:]:
            raise ValueError(f"Unknown side-feature set: {feature_set}")
        super().__init__(n_users, n_items, dim)
        self.feature_set = feature_set
        self.register_buffer("user_features", user_features)
        self.register_buffer("item_features", item_features)
        self.user_projection = (
            nn.Linear(user_features.shape[1], dim, bias=False)
            if feature_set in {"user", "both"} else None
        )
        self.item_projection = (
            nn.Linear(item_features.shape[1], dim, bias=False)
            if feature_set in {"item", "both"} else None
        )
        for projection in (self.user_projection, self.item_projection):
            if projection is not None:
                nn.init.zeros_(projection.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embedding(users)
        item_vectors = self.item_embedding(items)
        if self.user_projection is not None:
            user_vectors = user_vectors + self.user_projection(self.user_features[users])
        if self.item_projection is not None:
            item_vectors = item_vectors + self.item_projection(self.item_features[items])
        return (
            (user_vectors * item_vectors).sum(-1)
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
            + self.global_bias
        )


def run(output_dir: Path, epochs: int = 100, patience: int = 5) -> dict[str, object]:
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    project_dir = Path(__file__).resolve().parent
    config = Config(epochs=epochs, patience=patience, output_dir=str(output_dir))
    ratings = load_ratings(project_dir / "data/ml-100k/u.data")
    train, validation, test = temporal_split(ratings)
    user_features, item_features, feature_metadata = load_side_features(
        ratings, project_dir / "data/ml-100k/u.user", project_dir / "data/ml-100k/u.item"
    )
    known_items = set(train["item_idx"].astype(int))
    # This is the selected historical control: unlabeled, train-unobserved only.
    sampling_data = prepare_sampling(train, known_items, include_real_negatives=False)
    explicit = observed_binary(train)[:3]
    validation_data = build_ranking_data(validation, train, known_items)
    test_data = build_ranking_data(
        test, pd.concat((train, validation), ignore_index=True), known_items
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for feature_set in FEATURE_SETS:
        set_seed(config.seed)
        model = (
            MF(len(user_features), len(item_features), config.dim)
            if feature_set == "none"
            else FeatureMF(
                len(user_features), len(item_features), config.dim,
                user_features, item_features, feature_set,
            )
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        run_dir = output_dir / feature_set
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "best_model.pt"
        best_ndcg = -1.0
        best_epoch = 0
        stale = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, config.epochs + 1):
            train_loss, n_units = train_epoch(
                model, optimizer, "sampled", "bpr", train, known_items,
                explicit, sampling_data, epoch, config, device,
            )
            validation_scores = evaluate_ranking(
                model, validation_data, config.k, device, config.score_batch_size
            )
            ndcg = float(validation_scores[f"ndcg@{config.k}"])
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "training_units": n_units,
                f"validation_recall@{config.k}": validation_scores[f"recall@{config.k}"],
                f"validation_hit_ratio@{config.k}": validation_scores[f"hit_ratio@{config.k}"],
                f"validation_ndcg@{config.k}": ndcg,
            })
            if ndcg > best_ndcg:
                best_ndcg, best_epoch, stale = ndcg, epoch, 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0 or stale >= config.patience:
                print(
                    f"{feature_set} epoch={epoch:03d} val_ndcg={ndcg:.4f} "
                    f"best_epoch={best_epoch}", flush=True
                )
            if stale >= config.patience:
                break
        model.load_state_dict(torch.load(
            checkpoint_path, map_location=device, weights_only=True
        ))
        test_scores = evaluate_ranking(
            model, test_data, config.k, device, config.score_batch_size
        )
        result: dict[str, object] = {
            "feature_set": feature_set,
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "trained_epochs": len(history),
            "best_epoch": best_epoch,
            "best_validation": history[best_epoch - 1],
            "test": test_scores,
            "history": history,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        results.append({key: value for key, value in result.items() if key != "history"})
        print(json.dumps({
            "feature_set": feature_set, "best_epoch": best_epoch,
            "test_ndcg@10": test_scores["ndcg@10"],
        }), flush=True)

    summary: dict[str, object] = {
        "selection_rule": "highest prior validation nDCG@10 across 60-epoch original-pool and 100-epoch mixed-pool benchmarks",
        "control": "MF + BPR + one train-unobserved, globally train-known item per positive per epoch",
        "config": asdict(config),
        "sampling": "train-unobserved only; no real train negatives; no validation/test interactions inspected",
        "feature_metadata": feature_metadata,
        "feature_shapes": {
            "user": list(user_features.shape), "item": list(item_features.shape)
        },
        "split_counts": [len(train), len(validation), len(test)],
        "validation_users": validation_data.eligible_users,
        "test_users": test_data.eligible_users,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/side_features_mf_bpr_100"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    run(args.output_dir, args.epochs, args.patience)
