from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from train import MatrixFactorization, load_ratings, temporal_split


def load_titles(path: Path) -> dict[int, str]:
    movies = pd.read_csv(
        path,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"],
    )
    return dict(zip(movies["item_id"].astype(int), movies["title"].astype(str)))


@torch.no_grad()
def recommend(
    model: MatrixFactorization,
    user_idx: int,
    candidate_item_indices: list[int],
    top_k: int,
    device: torch.device,
) -> list[tuple[int, float]]:
    if not candidate_item_indices:
        return []
    model.eval()
    items = torch.tensor(candidate_item_indices, dtype=torch.long, device=device)
    users = torch.full_like(items, fill_value=user_idx)
    normalized_scores = model(users, items)
    k = min(top_k, len(candidate_item_indices))
    scores, positions = torch.topk(normalized_scores, k=k)
    return [
        (candidate_item_indices[position], float(score))
        for position, score in zip(positions.cpu().tolist(), scores.cpu().tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend Top-K MovieLens films for one user")
    parser.add_argument("--user-id", type=int, required=True, help="Original MovieLens user ID")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--data-path", default="data/ml-100k/u.data")
    parser.add_argument("--item-path", default="data/ml-100k/u.item")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--use-bias", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    project_dir = Path(__file__).resolve().parent
    ratings = load_ratings(project_dir / args.data_path)
    train, _, _ = temporal_split(ratings)

    user_rows = ratings.loc[ratings["user_id"] == args.user_id]
    if user_rows.empty:
        raise ValueError(f"Unknown MovieLens user_id: {args.user_id}")
    user_idx = int(user_rows["user_idx"].iloc[0])

    # At the evaluation cutoff, only training-period interactions are known.
    seen_by_user = set(
        train.loc[train["user_id"] == args.user_id, "item_idx"].astype(int).tolist()
    )
    globally_known_items = set(train["item_idx"].astype(int).unique().tolist())
    candidates = sorted(globally_known_items - seen_by_user)

    num_users = int(ratings["user_idx"].max()) + 1
    num_items = int(ratings["item_idx"].max()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatrixFactorization(
        num_users, num_items, args.embedding_dim, use_bias=args.use_bias
    ).to(device)
    state_dict = torch.load(
        project_dir / args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    ranked = recommend(model, user_idx, candidates, args.top_k, device)
    item_index_to_id = (
        ratings[["item_idx", "item_id"]]
        .drop_duplicates()
        .set_index("item_idx")["item_id"]
        .astype(int)
        .to_dict()
    )
    titles = load_titles(project_dir / args.item_path)

    print(
        f"Top-{len(ranked)} recommendations for user_id={args.user_id} "
        f"from {len(candidates)} warm unseen candidates"
    )
    print("rank\titem_id\tpred_rating\ttitle")
    for rank, (item_idx, normalized_score) in enumerate(ranked, start=1):
        item_id = item_index_to_id[item_idx]
        predicted_rating = 1.0 + 4.0 * normalized_score
        print(f"{rank}\t{item_id}\t{predicted_rating:.3f}\t{titles.get(item_id, '<unknown>')}")


if __name__ == "__main__":
    main()
