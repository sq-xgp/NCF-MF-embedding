"""Fuse the user's frozen Word2Vec title vectors into sampled-MSE MF and NCF.

Uses the exact fusion architectures and training/evaluation loop of the static
BERT experiment, with a 100-to-64 projection instead of 768-to-64. No BERT
weights or tokenizer are loaded, and no Transformer is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

from benchmark_bert_static_embeddings import run_one
from benchmark_ranking_losses import (
    Config,
    SAMPLING_PROTOCOL,
    build_ranking_data,
    observed_binary,
    prepare_sampling,
)
from train import load_ratings, temporal_split


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_title_vectors(
    path: Path, ratings: pd.DataFrame,
) -> tuple[torch.Tensor, dict]:
    """Align exported movie vectors by original item ID, never by row position."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected an exported dictionary with vectors and item_ids")
    vectors = payload["vectors"]
    item_ids = [int(item) for item in payload["item_ids"]]
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
        raise ValueError("Word2Vec vectors must be a two-dimensional tensor")
    if vectors.shape[0] != len(item_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError("Word2Vec rows and unique item IDs must match")
    if vectors.shape[1] <= 0 or not torch.isfinite(vectors).all():
        raise ValueError("Invalid Word2Vec dimensions or non-finite values")
    row_by_id = {item: row for row, item in enumerate(item_ids)}
    if "item_id_to_idx" in payload:
        declared = {int(k): int(v) for k, v in payload["item_id_to_idx"].items()}
        if declared != row_by_id:
            raise ValueError("The exported item ID map disagrees with item_ids")
    item_order = (
        ratings[["item_id", "item_idx"]].drop_duplicates()
        .sort_values("item_idx")["item_id"].astype(int).tolist()
    )
    missing = sorted(set(item_order) - row_by_id.keys())
    if missing:
        raise ValueError(f"No Word2Vec title vectors for movie IDs: {missing[:20]}")
    aligned = vectors[[row_by_id[item] for item in item_order]].detach().float().contiguous()
    norms = aligned.norm(dim=1)
    coverage_path = path.with_name("coverage.json")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {}
    if coverage:
        if coverage["embedding_dim"] != aligned.shape[1] or coverage["movie_count"] != len(item_order):
            raise ValueError("Coverage metadata disagrees with the supplied title vectors")
        if coverage["covered_movies"] != int((norms > 0).sum()):
            raise ValueError("Coverage metadata disagrees with the number of nonzero movie vectors")
    metadata = {
        "model": "user-trained Word2Vec",
        "embedding_dim": aligned.shape[1],
        "title_count": len(item_order),
        "title_pooling": payload.get("aggregation", "provided precomputed title vectors"),
        "frozen": True,
        "normalization": "none; use the supplied exported vector values unchanged",
        "alignment": "original MovieLens item_id mapped to shared ratings item_idx",
        "source_file": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "source_sha256": digest(path),
        "source_checkpoint": payload.get("source_checkpoint"),
        "source_training_config": payload.get("config", {}),
        "coverage": coverage,
        "coverage_sha256": digest(coverage_path) if coverage else None,
        "nonzero_movie_vectors": int((norms > 0).sum()),
        "vector_norm_min": float(norms.min()),
        "vector_norm_mean": float(norms.mean()),
        "vector_norm_max": float(norms.max()),
        "pretraining_corpus": "not established from the provided export metadata",
    }
    return aligned, metadata


def population(data) -> dict:
    return {
        "eligible_users": data.eligible_users,
        "skipped_users": data.skipped_users,
        "cold_positives": data.cold_positives,
        "total_candidates": data.total_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=Path("data/pretrained_word2vec/movielens_title_embeddings.pt"))
    parser.add_argument("--output-dir", default="outputs/word2vec_title_mixed_pool_100")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    source = (ROOT / args.embeddings).resolve()
    output_root = ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    ratings_path = ROOT / "data/ml-100k/u.data"
    ratings = load_ratings(ratings_path)
    title_vectors, metadata = load_title_vectors(source, ratings)
    torch.save({"embeddings": title_vectors, "metadata": metadata}, output_root / "title_embeddings.pt")
    train, validation, test = temporal_split(ratings)
    known_items = set(train["item_idx"].astype(int))
    explicit = observed_binary(train)[:3]
    sampling_data = prepare_sampling(train, known_items)
    validation_data = build_ranking_data(validation, train, known_items)
    test_history = pd.concat((train, validation), ignore_index=True)
    test_data = build_ranking_data(test, test_history, known_items)
    config = Config(epochs=args.epochs, patience=args.patience, output_dir=args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device": str(device), "word2vec": metadata}, ensure_ascii=False), flush=True)
    results = [
        run_one(
            name, title_vectors, metadata, config, train, explicit, sampling_data,
            known_items, validation_data, test_data,
            int(ratings["user_idx"].max()) + 1, device, output_root,
            feature="frozen user-trained Word2Vec title vectors; mean of in-vocabulary center word vectors; no BERT",
            feature_metadata_key="word2vec",
        )
        for name in ("mf", "ncf")
    ]
    summary = {
        "config": asdict(config),
        "selection_basis": "sampled-MSE had highest validation nDCG@10 within both MF and NCF in the original 10-run benchmark",
        "sampling_protocol": SAMPLING_PROTOCOL,
        "protocol": {
            "training": "same sampled-MSE negative pairs and per-user temporal split as the original benchmark",
            "checkpoint_selection": "highest validation nDCG@10; earliest epoch for ties; early stopping patience",
            "test_usage": "once after checkpoint selection",
            "title_features": "user-provided frozen 100D Word2Vec movie vectors, pooled over matched title words",
            "fusion": "trainable bias-free 100-to-64 linear projection added to trainable movie ID embedding",
            "cold_items": "original ranking evaluation includes train-known candidate items only",
            "comparability": "same training and scoring protocol; different feature dimensions imply different projection parameter counts and RNG initialization trajectories",
        },
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "validation_population": population(validation_data),
        "test_population": population(test_data),
        "ratings_sha256": digest(ratings_path),
        "runtime": {"torch": str(torch.__version__), "device": str(device),
                    "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None},
        "results": [{key: value for key, value in result.items() if key != "history"} for result in results],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed": [{"model": r["model"], "best_epoch": r["best_epoch"], "test": r["test"]} for r in results]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
