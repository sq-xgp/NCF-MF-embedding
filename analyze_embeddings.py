from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from train import load_ratings, temporal_split
from train_bpr import BPRModel


GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_movies(path: Path) -> pd.DataFrame:
    names = ["item_id", "title", "release_date", "video_release_date", "url", *GENRES]
    movies = pd.read_csv(path, sep="|", encoding="latin-1", header=None, names=names)
    movies["genres"] = movies.apply(
        lambda row: [genre for genre in GENRES if int(row[genre]) == 1], axis=1
    )
    return movies[["item_id", "title", "genres"]]


def load_users(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )


def cosine_neighbors(matrix: torch.Tensor, index: int, top_k: int) -> list[tuple[int, float]]:
    normalized = F.normalize(matrix, dim=1)
    similarities = normalized @ normalized[index]
    similarities[index] = -float("inf")
    values, indices = torch.topk(similarities, k=min(top_k, len(matrix) - 1))
    return list(zip(indices.tolist(), values.tolist()))


def top_genres(
    user_id: int,
    train: pd.DataFrame,
    movie_by_id: dict[int, dict[str, object]],
    limit: int = 3,
) -> list[str]:
    liked = train.loc[(train["user_id"] == user_id) & (train["rating"] >= 4)]
    counts: Counter[str] = Counter()
    for item_id in liked["item_id"].astype(int):
        counts.update(movie_by_id[item_id]["genres"])
    return [genre for genre, _ in counts.most_common(limit)]


def pca_2d(matrix: torch.Tensor) -> torch.Tensor:
    centered = matrix.float().cpu() - matrix.float().cpu().mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(centered, q=2)
    return centered @ components[:, :2]


def create_plots(
    output_dir: Path,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    train: pd.DataFrame,
    selected_user_idx: int,
    selected_item_idx: int,
    similar_user_indices: list[int],
    similar_item_indices: list[int],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    user_xy = pca_2d(user_embeddings).numpy()
    item_xy = pca_2d(item_embeddings).numpy()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(user_xy[:, 0], user_xy[:, 1], s=10, alpha=0.25, color="#4C78A8")
    highlighted_users = [selected_user_idx, *similar_user_indices]
    ax.scatter(
        user_xy[highlighted_users, 0], user_xy[highlighted_users, 1],
        s=70, color="#E45756", edgecolor="black",
    )
    for idx in highlighted_users:
        ax.annotate(f"user {idx + 1}", user_xy[idx], xytext=(5, 5), textcoords="offset points")
    ax.set(title="User embeddings: PCA projection", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=0.2)
    path = output_dir / "user_embedding_pca.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(item_xy[:, 0], item_xy[:, 1], s=10, alpha=0.2, color="#72B7B2")
    highlighted_items = [selected_item_idx, *similar_item_indices]
    ax.scatter(
        item_xy[highlighted_items, 0], item_xy[highlighted_items, 1],
        s=70, color="#F58518", edgecolor="black",
    )
    item_id_by_idx = (
        train[["item_idx", "item_id"]].drop_duplicates().set_index("item_idx")["item_id"].to_dict()
    )
    title_by_id = movies.set_index("item_id")["title"].to_dict()
    for idx in highlighted_items:
        item_id = int(item_id_by_idx.get(idx, idx + 1))
        ax.annotate(title_by_id.get(item_id, str(item_id))[:24], item_xy[idx], xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(title="Item embeddings: PCA projection", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=0.2)
    path = output_dir / "item_embedding_pca.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    item_counts = train.groupby("item_idx").size().reindex(range(len(item_embeddings)), fill_value=0)
    item_norms = item_embeddings.norm(dim=1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(item_counts.to_numpy(), item_norms, s=12, alpha=0.35, color="#54A24B")
    ax.set_xscale("log")
    ax.set(title="Embedding norm vs. training popularity", xlabel="Training ratings (log scale)", ylabel="Item embedding L2 norm")
    ax.grid(alpha=0.2)
    path = output_dir / "item_norm_vs_popularity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="BPR embedding case studies")
    parser.add_argument("--checkpoint", default="outputs/bpr/best_model.pt")
    parser.add_argument("--data-dir", default="data/ml-100k")
    parser.add_argument("--output-dir", default="outputs/embedding_analysis")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--item-id", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / args.data_dir
    output_dir = project_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ratings = load_ratings(data_dir / "u.data")
    train, validation, test = temporal_split(ratings)
    movies = load_movies(data_dir / "u.item")
    users = load_users(data_dir / "u.user")
    movie_by_id = movies.set_index("item_id").to_dict("index")
    user_by_id = users.set_index("user_id").to_dict("index")
    user_id_to_idx = ratings[["user_id", "user_idx"]].drop_duplicates().set_index("user_id")["user_idx"].astype(int).to_dict()
    item_id_to_idx = ratings[["item_id", "item_idx"]].drop_duplicates().set_index("item_id")["item_idx"].astype(int).to_dict()
    user_idx_to_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
    item_idx_to_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
    if args.user_id not in user_id_to_idx or args.item_id not in item_id_to_idx:
        raise ValueError("Unknown user-id or item-id")

    metrics = json.loads((project_dir / args.checkpoint).with_name("metrics.json").read_text(encoding="utf-8"))
    dim = int(metrics["config"]["embedding_dim"])
    model = BPRModel(len(user_id_to_idx), len(item_id_to_idx), dim)
    model.load_state_dict(torch.load(project_dir / args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    user_embeddings = model.user_embedding.weight.detach()
    item_embeddings = model.item_embedding.weight.detach()
    selected_user_idx = user_id_to_idx[args.user_id]
    selected_item_idx = item_id_to_idx[args.item_id]

    similar_users_raw = cosine_neighbors(user_embeddings, selected_user_idx, args.top_k)
    similar_users = []
    selected_genres = top_genres(args.user_id, train, movie_by_id)
    for idx, similarity in similar_users_raw:
        user_id = user_idx_to_id[idx]
        profile = user_by_id[user_id]
        similar_users.append({
            "user_id": user_id,
            "cosine_similarity": similarity,
            "age": int(profile["age"]),
            "gender": profile["gender"],
            "occupation": profile["occupation"],
            "top_liked_genres_in_train": top_genres(user_id, train, movie_by_id),
        })

    similar_items_raw = cosine_neighbors(item_embeddings, selected_item_idx, args.top_k)
    similar_items = []
    selected_item_genres = set(movie_by_id[args.item_id]["genres"])
    for idx, similarity in similar_items_raw:
        item_id = item_idx_to_id[idx]
        genres = movie_by_id[item_id]["genres"]
        similar_items.append({
            "item_id": item_id,
            "title": movie_by_id[item_id]["title"],
            "cosine_similarity": similarity,
            "genres": genres,
            "shared_genres": sorted(selected_item_genres & set(genres)),
        })

    known_items = set(train["item_idx"].astype(int).unique())
    user_seen = set(train.loc[train["user_id"] == args.user_id, "item_idx"].astype(int))
    candidate_items = sorted(known_items - user_seen)
    with torch.no_grad():
        item_tensor = torch.tensor(candidate_items)
        user_tensor = torch.full_like(item_tensor, selected_user_idx)
        scores = model.score(user_tensor, item_tensor)
        values, positions = torch.topk(scores, k=min(args.top_k, len(candidate_items)))
    later_user_ratings = pd.concat([validation, test]).loc[
        lambda df: df["user_id"] == args.user_id
    ].set_index("item_id")["rating"].to_dict()
    preferred_items = []
    for position, score in zip(positions.tolist(), values.tolist()):
        item_id = item_idx_to_id[candidate_items[position]]
        preferred_items.append({
            "item_id": item_id,
            "title": movie_by_id[item_id]["title"],
            "bpr_score": score,
            "genres": movie_by_id[item_id]["genres"],
            "later_observed_rating": later_user_ratings.get(item_id),
        })

    item_seen_users = set(train.loc[train["item_id"] == args.item_id, "user_idx"].astype(int))
    candidate_users = sorted(set(train["user_idx"].astype(int).unique()) - item_seen_users)
    later_item_ratings = pd.concat([validation, test]).loc[
        lambda df: df["item_id"] == args.item_id
    ].set_index("user_id")["rating"].to_dict()
    # Raw BPR scores are identifiable only within a user, not across users. Rank the
    # fixed item inside each user's own warm unseen candidate set, then compare ranks.
    user_preference_rows = []
    for user_idx in candidate_users:
        user_id = user_idx_to_id[user_idx]
        seen = set(train.loc[train["user_id"] == user_id, "item_idx"].astype(int))
        user_candidates = sorted(known_items - seen)
        if selected_item_idx not in user_candidates:
            continue
        with torch.no_grad():
            items = torch.tensor(user_candidates)
            user_tensor = torch.full_like(items, user_idx)
            candidate_scores = model.score(user_tensor, items)
        target_position = user_candidates.index(selected_item_idx)
        target_score = float(candidate_scores[target_position])
        within_user_rank = 1 + int((candidate_scores > target_score).sum())
        preference_percentile = 1.0 - (within_user_rank - 1) / max(
            len(user_candidates) - 1, 1
        )
        score_std = float(candidate_scores.std(unbiased=False))
        within_user_zscore = (
            (target_score - float(candidate_scores.mean())) / score_std
            if score_std > 0
            else 0.0
        )
        user_preference_rows.append(
            (within_user_zscore, preference_percentile, -within_user_rank, target_score, user_idx, len(user_candidates))
        )
    user_preference_rows.sort(reverse=True)
    preferred_users = []
    for zscore, percentile, negative_rank, score, user_idx, candidate_count in user_preference_rows[: args.top_k]:
        user_id = user_idx_to_id[user_idx]
        profile = user_by_id[user_id]
        preferred_users.append({
            "user_id": user_id,
            "within_user_rank": -negative_rank,
            "candidate_count": candidate_count,
            "preference_percentile": percentile,
            "within_user_zscore": zscore,
            "raw_bpr_score": score,
            "age": int(profile["age"]),
            "gender": profile["gender"],
            "occupation": profile["occupation"],
            "top_liked_genres_in_train": top_genres(user_id, train, movie_by_id),
            "later_observed_rating": later_item_ratings.get(user_id),
        })

    plot_paths = create_plots(
        output_dir,
        user_embeddings,
        item_embeddings,
        users,
        movies,
        train,
        selected_user_idx,
        selected_item_idx,
        [idx for idx, _ in similar_users_raw],
        [idx for idx, _ in similar_items_raw],
    )
    selected_profile = user_by_id[args.user_id]
    result = {
        "model": "BPR MF best checkpoint",
        "similarity_metric": "cosine similarity over embedding vectors; bias excluded",
        "selected_user": {
            "user_id": args.user_id,
            "age": int(selected_profile["age"]),
            "gender": selected_profile["gender"],
            "occupation": selected_profile["occupation"],
            "top_liked_genres_in_train": selected_genres,
        },
        "similar_users": similar_users,
        "selected_item": {
            "item_id": args.item_id,
            "title": movie_by_id[args.item_id]["title"],
            "genres": movie_by_id[args.item_id]["genres"],
        },
        "similar_items": similar_items,
        "preferred_unseen_items_for_user": preferred_items,
        "preferred_unseen_users_for_item": preferred_users,
        "visualizations": plot_paths,
    }
    json_path = output_dir / "case_studies.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Embedding Case Studies", "",
        f"Model: BPR MF; cosine similarity excludes the item bias term.", "",
        f"## User {args.user_id}", "",
        f"Profile: age {selected_profile['age']}, {selected_profile['gender']}, {selected_profile['occupation']}; top liked genres in train: {', '.join(selected_genres)}.", "",
        "### Similar users", "",
    ]
    for row in similar_users:
        lines.append(f"- User {row['user_id']}: cosine={row['cosine_similarity']:.3f}; profile={row['age']}/{row['gender']}/{row['occupation']}; liked genres={', '.join(row['top_liked_genres_in_train'])}.")
    lines += ["", f"## Item {args.item_id}: {movie_by_id[args.item_id]['title']}", "", f"Genres: {', '.join(movie_by_id[args.item_id]['genres'])}.", "", "### Similar items", ""]
    for row in similar_items:
        lines.append(f"- {row['item_id']} {row['title']}: cosine={row['cosine_similarity']:.3f}; genres={', '.join(row['genres'])}; shared={', '.join(row['shared_genres']) or 'none'}.")
    lines += ["", f"## Preferred items not in user {args.user_id}'s training history", ""]
    for row in preferred_items:
        later = row["later_observed_rating"] if row["later_observed_rating"] is not None else "not later observed"
        lines.append(f"- {row['item_id']} {row['title']}: BPR score={row['bpr_score']:.3f}; genres={', '.join(row['genres'])}; later rating={later}.")
    lines += ["", f"## Users predicted to prefer item {args.item_id} but absent from its training interactions", "", "Users are ordered by the item's percentile inside each user's own candidate ranking; raw BPR scores are not compared across users.", ""]
    for row in preferred_users:
        later = row["later_observed_rating"] if row["later_observed_rating"] is not None else "not later observed"
        lines.append(f"- User {row['user_id']}: within-user z={row['within_user_zscore']:.3f}; item rank={row['within_user_rank']}/{row['candidate_count']}; percentile={row['preference_percentile']:.3f}; raw score={row['raw_bpr_score']:.3f}; profile={row['age']}/{row['gender']}/{row['occupation']}; liked genres={', '.join(row['top_liked_genres_in_train'])}; later rating={later}.")
    lines += ["", "## Interpretation cautions", "", "- Cosine neighbors show learned behavioral similarity, not causal or demographic equivalence.", "- A BPR score is meaningful for ordering items within one user; raw score magnitudes are not calibrated across users.", "- A high BPR score is not a probability or predicted 1--5 rating.", "- PCA is a lossy 2D projection; distance in the plot is not identical to 32D cosine similarity.", "- Embedding norms can encode activity/popularity and should be inspected alongside cosine similarity."]
    markdown_path = output_dir / "case_studies.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "report": str(markdown_path), "plots": plot_paths}, indent=2))


if __name__ == "__main__":
    main()
