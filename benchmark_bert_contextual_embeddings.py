"""Extract frozen BERT sentence embeddings with Transformers, then train MF/NCF."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

from bert_embeddings import default_model_dir, extract_bert_embeddings, movie_titles
from benchmark_bert_static_embeddings import run_one
from benchmark_ranking_losses import Config, build_ranking_data, observed_binary, prepare_sampling
from train import load_ratings, temporal_split


def encode_titles(
    project_dir: Path, ratings: pd.DataFrame, model_dir: Path,
    device: torch.device, batch_size: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    return extract_bert_embeddings(
        movie_titles(project_dir, ratings), model_dir,
        mode="sentence", device=device, batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert-model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", default="outputs/bert_contextual_title_transformers_100")
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
    title_vectors, model_info = encode_titles(
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
    config = Config(epochs=args.epochs, patience=args.patience, output_dir=args.output_dir)
    info = {**model_info, "model_snapshot": model_dir.name}
    print(json.dumps({"device": str(device), "bert": info}), flush=True)
    results = [
        run_one(
            name, title_vectors, info, config, train, explicit, sampling_data,
            known_items, validation_data, test_data,
            int(ratings["user_idx"].max()) + 1, device, output_root,
            feature="frozen bert-base-uncased final-layer title token states",
        )
        for name in ("mf", "ncf")
    ]
    summary = {
        "config": asdict(config),
        "selection_basis": "sampled-MSE was highest by validation nDCG@10 within MF and NCF in the original 10-run benchmark",
        "protocol": {
            "training": "same sampled-MSE negative pairs and temporal split as original ranking benchmark",
            "checkpoint_selection": "highest validation nDCG@10; early stopping patience",
            "test_usage": "once after checkpoint selection",
            "title_features": "frozen BERT final-layer contextual token outputs, mean-pooled per title",
            "cold_items": "original ranking candidate set includes train-known items only",
        },
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "results": [{key: value for key, value in result.items() if key != "history"} for result in results],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "completed": [
            f"{result['model']}: val_nDCG@10={result['best_validation_ndcg']:.4f}, "
            f"test_nDCG@10={result['test']['ndcg@10']:.4f}"
            for result in results
        ]
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
