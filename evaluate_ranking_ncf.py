from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_ranking import recall_ndcg_at_k
from recommend import recommend
from train import load_ratings, temporal_split
from train_ncf import NeuralCollaborativeFiltering


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate warm-item Top-K ranking for NCF")
    parser.add_argument("--checkpoint", default="outputs/ncf/best_model.pt")
    parser.add_argument("--data-path", default="data/ml-100k/u.data")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--positive-threshold", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if not 1 <= args.positive_threshold <= 5:
        raise ValueError("--positive-threshold must be between 1 and 5")

    project_dir = Path(__file__).resolve().parent
    checkpoint_path = project_dir / args.checkpoint
    training_results = json.loads(
        (checkpoint_path.parent / "metrics.json").read_text(encoding="utf-8")
    )
    config = training_results["config"]

    ratings = load_ratings(project_dir / args.data_path)
    train, validation, test = temporal_split(ratings)
    known_items = set(train["item_idx"].astype(int).unique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralCollaborativeFiltering(
        num_users=int(ratings["user_idx"].max()) + 1,
        num_items=int(ratings["item_idx"].max()) + 1,
        embedding_dim=int(config["embedding_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )

    recalls: list[float] = []
    ndcgs: list[float] = []
    skipped_no_positive = 0
    cold_positive_count = 0

    for user_id, test_rows in test.groupby("user_id", sort=True):
        user_idx = int(test_rows["user_idx"].iloc[0])
        positive_mask = test_rows["rating"] >= args.positive_threshold
        warm_mask = test_rows["item_idx"].isin(known_items)
        relevant = set(
            test_rows.loc[positive_mask & warm_mask, "item_idx"].astype(int).tolist()
        )
        cold_positive_count += int((positive_mask & ~warm_mask).sum())
        if not relevant:
            skipped_no_positive += 1
            continue

        # Validation interactions precede test interactions and are known at cutoff.
        train_seen = train.loc[train["user_id"] == user_id, "item_idx"]
        validation_seen = validation.loc[validation["user_id"] == user_id, "item_idx"]
        seen = set(train_seen.astype(int)) | set(validation_seen.astype(int))
        candidates = sorted(known_items - seen)
        ranked = recommend(model, user_idx, candidates, args.k, device)
        recommended_items = [item_idx for item_idx, _ in ranked]
        recall, ndcg = recall_ndcg_at_k(recommended_items, relevant, args.k)
        recalls.append(recall)
        ndcgs.append(ndcg)

    if not recalls:
        raise RuntimeError("No users have warm positive test items")
    result = {
        "model": "single_tower_ncf_explicit_mse",
        "checkpoint": str(checkpoint_path.relative_to(project_dir)),
        "protocol": {
            "k": args.k,
            "positive_rating_threshold": args.positive_threshold,
            "candidate_items": "items seen globally in train",
            "excluded_user_history": "train + validation interactions",
            "aggregation": "macro average over users with >=1 warm positive test item",
        },
        "evaluated_users": len(recalls),
        "skipped_users_without_warm_positive": skipped_no_positive,
        "excluded_cold_positive_count": cold_positive_count,
        f"recall@{args.k}": sum(recalls) / len(recalls),
        f"ndcg@{args.k}": sum(ndcgs) / len(ndcgs),
        "training_objective_note": "NCF was trained with explicit-rating MSE, not a ranking loss",
    }
    output_path = (
        project_dir / args.output
        if args.output
        else checkpoint_path.parent / "ranking_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved ranking metrics to {output_path}")


if __name__ == "__main__":
    main()
