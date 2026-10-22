from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from recommend import recommend
from train import MatrixFactorization, load_ratings, temporal_split


def recall_ndcg_at_k(
    recommended: list[int], relevant: set[int], k: int
) -> tuple[float, float]:
    if not relevant:
        raise ValueError("relevant must contain at least one item")
    ranked = recommended[:k]
    hits = sum(item in relevant for item in ranked)
    recall = hits / len(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked, start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return recall, dcg / idcg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate warm-item Top-K ranking")
    parser.add_argument("--checkpoint", default="outputs/biased_tuned/best_model.pt")
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
    training_metrics_path = checkpoint_path.parent / "metrics.json"
    training_results = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    config = training_results["config"]

    ratings = load_ratings(project_dir / args.data_path)
    train, validation, test = temporal_split(ratings)
    known_items = set(train["item_idx"].astype(int).unique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatrixFactorization(
        num_users=int(ratings["user_idx"].max()) + 1,
        num_items=int(ratings["item_idx"].max()) + 1,
        embedding_dim=int(config["embedding_dim"]),
        use_bias=bool(config["use_bias"]),
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
        warm_positive_rows = test_rows.loc[
            (test_rows["rating"] >= args.positive_threshold)
            & (test_rows["item_idx"].isin(known_items))
        ]
        relevant = set(warm_positive_rows["item_idx"].astype(int).tolist())
        cold_positive_count += int(
            (
                (test_rows["rating"] >= args.positive_threshold)
                & (~test_rows["item_idx"].isin(known_items))
            ).sum()
        )
        if not relevant:
            skipped_no_positive += 1
            continue

        # Validation interactions precede test interactions, so they are known history
        # at the test cutoff even though they were not used for gradient updates.
        train_seen = train.loc[train["user_id"] == user_id, "item_idx"]
        validation_seen = validation.loc[validation["user_id"] == user_id, "item_idx"]
        seen = set(train_seen.astype(int).tolist()) | set(
            validation_seen.astype(int).tolist()
        )
        candidates = sorted(known_items - seen)
        ranked = recommend(model, user_idx, candidates, args.k, device)
        recommended_items = [item_idx for item_idx, _ in ranked]
        recall, ndcg = recall_ndcg_at_k(recommended_items, relevant, args.k)
        recalls.append(recall)
        ndcgs.append(ndcg)

    if not recalls:
        raise RuntimeError("No users have warm positive test items")
    result = {
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
