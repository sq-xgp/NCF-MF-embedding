"""Add frozen BERT word-table title features to the best MF and NCF rankers.

The title representation is the mean of the pretrained BERT input word
embeddings for its WordPiece tokens. BERT's Transformer layers are not run.
Both rankers use the sampled-MSE protocol selected by validation nDCG@10 in
the original 10-run benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn

from benchmark_ranking_losses import (
    Config,
    SAMPLING_PROTOCOL,
    build_ranking_data,
    evaluate_ranking,
    observed_binary,
    prepare_sampling,
    train_epoch,
)
from train import load_ratings, set_seed, temporal_split
from bert_embeddings import default_model_dir, extract_bert_embeddings, movie_titles


def build_title_embeddings(
    project_dir: Path, ratings: pd.DataFrame, model_dir: Path,
    device: torch.device | str = "cpu", batch_size: int = 32,
) -> tuple[torch.Tensor, dict[str, object]]:
    return extract_bert_embeddings(
        movie_titles(project_dir, ratings), model_dir,
        mode="word", device=device, batch_size=batch_size,
    )


class StaticTitleMF(nn.Module):
    def __init__(self, n_users: int, title_vectors: torch.Tensor, dim: int) -> None:
        super().__init__()
        n_items = title_vectors.shape[0]
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_id_embedding = nn.Embedding(n_items, dim)
        self.title_projection = nn.Linear(title_vectors.shape[1], dim, bias=False)
        self.register_buffer("title_vectors", title_vectors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_id_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        item_vectors = self.item_id_embedding(items) + self.title_projection(self.title_vectors[items])
        return (
            (self.user_embedding(users) * item_vectors).sum(-1)
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
            + self.global_bias
        )


class StaticTitleNCF(nn.Module):
    def __init__(self, n_users: int, title_vectors: torch.Tensor, dim: int) -> None:
        super().__init__()
        n_items = title_vectors.shape[0]
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_id_embedding = nn.Embedding(n_items, dim)
        self.title_projection = nn.Linear(title_vectors.shape[1], dim, bias=False)
        self.register_buffer("title_vectors", title_vectors)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(),
            nn.Linear(dim, dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_id_embedding.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        item_vectors = self.item_id_embedding(items) + self.title_projection(self.title_vectors[items])
        return self.mlp(torch.cat((self.user_embedding(users), item_vectors), dim=-1)).squeeze(-1)


def run_one(
    model_name: str,
    title_vectors: torch.Tensor,
    model_info: dict[str, object],
    config: Config,
    train: pd.DataFrame,
    explicit: tuple[np.ndarray, np.ndarray, np.ndarray],
    sampling_data,
    known_items: set[int],
    validation_data,
    test_data,
    n_users: int,
    device: torch.device,
    output_root: Path,
    *,
    feature: str = "frozen bert-base-uncased input word embedding table; mean WordPiece title vector",
    feature_metadata_key: str = "bert",
) -> dict[str, object]:
    set_seed(config.seed)
    model_type = StaticTitleMF if model_name == "mf" else StaticTitleNCF
    model = model_type(n_users, title_vectors, config.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float | int]] = []
    best_ndcg = -1.0
    best_epoch = 0
    stale = 0
    run_dir = output_root / f"{model_name}_sampled_mse"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss, n_units = train_epoch(
            model, optimizer, "sampled", "mse", train, known_items, explicit,
            sampling_data, epoch, config, device,
        )
        validation = evaluate_ranking(
            model, validation_data, config.k, device, config.score_batch_size,
        )
        ndcg = float(validation[f"ndcg@{config.k}"])
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "training_units": n_units,
            f"validation_recall@{config.k}": validation[f"recall@{config.k}"],
            f"validation_hit_ratio@{config.k}": validation[f"hit_ratio@{config.k}"],
            f"validation_ndcg@{config.k}": ndcg,
        })
        if ndcg > best_ndcg:
            best_ndcg, best_epoch, stale = ndcg, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= config.patience:
            print(
                f"{model_name}/sampled/mse epoch={epoch:03d} "
                f"loss={train_loss:.4f} val_ndcg={ndcg:.4f} best_epoch={best_epoch}",
                flush=True,
            )
        if stale >= config.patience:
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_metrics = evaluate_ranking(model, test_data, config.k, device, config.score_batch_size)
    result: dict[str, object] = {
        "model": model_name,
        "regime": "sampled",
        "loss": "mse",
        "sampling_protocol": SAMPLING_PROTOCOL,
        "feature": feature,
        "fusion": f"trainable linear projection to {config.dim} dimensions added to trainable item ID embedding",
        feature_metadata_key: model_info,
        "config": asdict(config),
        "parameter_count_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best_epoch,
        "best_validation_ndcg": best_ndcg,
        "best_validation": history[best_epoch - 1],
        "test": test_metrics,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert-model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", default="outputs/bert_static_word_embedding_transformers_100")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size-encode", type=int, default=32)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size_encode) <= 0:
        raise ValueError("epochs, patience, and batch-size-encode must be positive")
    project_dir = Path(__file__).resolve().parent
    output_root = project_dir / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = args.bert_model_dir or default_model_dir()
    ratings = load_ratings(project_dir / "data/ml-100k/u.data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    title_vectors, model_info = build_title_embeddings(
        project_dir, ratings, model_dir, device, args.batch_size_encode,
    )
    torch.save({"embeddings": title_vectors, "metadata": model_info}, output_root / "title_embeddings.pt")
    train, validation, test = temporal_split(ratings)
    known_items = set(train["item_idx"].astype(int))
    explicit = observed_binary(train)[:3]
    sampling_data = prepare_sampling(train, known_items)
    validation_data = build_ranking_data(validation, train, known_items)
    test_history = pd.concat((train, validation), ignore_index=True)
    test_data = build_ranking_data(test, test_history, known_items)
    print(json.dumps({"device": str(device), "bert": {k: v for k, v in model_info.items() if k != "vocab"}}), flush=True)
    config = Config(epochs=args.epochs, patience=args.patience, output_dir=args.output_dir)
    results = [
        run_one(
            model_name, title_vectors, model_info, config, train, explicit, sampling_data,
            known_items, validation_data, test_data,
            int(ratings["user_idx"].max()) + 1, device, output_root,
        )
        for model_name in ("mf", "ncf")
    ]
    summary = {
        "config": asdict(config),
        "selection_basis": "sampled-MSE configuration had the highest validation nDCG@10 within both MF and NCF in the original 10-run benchmark",
        "protocol": {
            "training": "same sampled-MSE negative pairs and temporal split as the original ranking benchmark",
            "checkpoint_selection": "highest validation nDCG@10; early stopping patience",
            "test_usage": "once after checkpoint selection",
            "title_features": "mean of static BERT input WordPiece embeddings; no Transformer layers",
            "cold_items": "title features are available for all items, but original ranking benchmark evaluates only train-known candidate items",
        },
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "results": [{key: val for key, val in result.items() if key != "history"} for result in results],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed": [f"{r['model']}: val_nDCG@10={r['best_validation_ndcg']:.4f}, test={r['test']}" for r in results]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
