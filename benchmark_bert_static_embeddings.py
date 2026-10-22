"""Add frozen BERT word-table title features to the best MF and NCF rankers.

The title representation is the mean of the pretrained BERT input word
embeddings for its WordPiece tokens. BERT's Transformer layers are not run.
Both rankers use the sampled-MSE protocol selected by validation nDCG@10 in
the original 10-run benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import unicodedata
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


MODEL_NAME = "bert-base-uncased"


def default_model_dir() -> Path:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snapshots = hub / "models--bert-base-uncased" / "snapshots"
    if snapshots.exists():
        available = sorted(
            (path for path in snapshots.iterdir() if (path / "model.safetensors").exists()),
            key=lambda path: path.name,
        )
        if available:
            return available[-1]
    return snapshots


def read_word_embedding_table(model_dir: Path) -> tuple[torch.Tensor, dict[str, object]]:
    """Read only embeddings.word_embeddings.weight from a BERT safetensors file."""
    model_file = model_dir / "model.safetensors"
    vocab_file = model_dir / "vocab.txt"
    config_file = model_dir / "config.json"
    for required in (model_file, vocab_file, config_file):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Download/cache {MODEL_NAME} or pass --bert-model-dir."
            )
    with model_file.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"Invalid safetensors header: {model_file}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size))
        key = "embeddings.word_embeddings.weight"
        if key not in header:
            key = "bert.embeddings.word_embeddings.weight"
        if key not in header:
            raise KeyError(f"No BERT word embedding tensor found in {model_file}")
        entry = header[key]
        if entry["dtype"] != "F32":
            raise TypeError(f"Expected float32 BERT word embeddings, got {entry['dtype']}")
        shape = tuple(int(value) for value in entry["shape"])
        start, end = (int(value) for value in entry["data_offsets"])
        handle.seek(8 + header_size + start)
        raw = handle.read(end - start)
    expected_bytes = math.prod(shape) * np.dtype("<f4").itemsize
    if len(raw) != expected_bytes:
        raise ValueError(f"Truncated BERT word embedding tensor: expected {expected_bytes} bytes")
    table = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    model_config = json.loads(config_file.read_text(encoding="utf-8"))
    vocab = vocab_file.read_text(encoding="utf-8").splitlines()
    if len(vocab) != shape[0]:
        raise ValueError(f"Vocabulary has {len(vocab)} tokens but embedding table has {shape[0]} rows")
    metadata = {
        "model": MODEL_NAME,
        "model_snapshot": model_dir.name,
        "embedding_tensor": key,
        "vocab_size": len(vocab),
        "embedding_dim": shape[1],
        "bert_hidden_size": model_config.get("hidden_size"),
    }
    return torch.from_numpy(table), {**metadata, "vocab": vocab}


def basic_tokens(text: str) -> list[str]:
    """Small uncased BERT BasicTokenizer equivalent for English movie titles."""
    cleaned = []
    for char in text:
        codepoint = ord(char)
        if codepoint in (0, 0xFFFD) or (codepoint < 32 and char not in "\t\n\r"):
            continue
        if char.isspace():
            cleaned.append(" ")
        else:
            cleaned.append(char)
    normalized = unicodedata.normalize("NFD", "".join(cleaned).lower())
    text = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    tokens: list[str] = []
    current = []
    for char in text:
        category = unicodedata.category(char)
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        elif category.startswith("P") or category.startswith("S"):
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(char)
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def wordpiece_ids(text: str, vocab: dict[str, int], unk_id: int) -> tuple[list[int], int]:
    ids: list[int] = []
    unknown = 0
    for token in basic_tokens(text):
        if len(token) > 100:
            ids.append(unk_id)
            unknown += 1
            continue
        start = 0
        pieces: list[str] = []
        while start < len(token):
            end = len(token)
            piece = None
            while start < end:
                candidate = token[start:end] if start == 0 else "##" + token[start:end]
                if candidate in vocab:
                    piece = candidate
                    break
                end -= 1
            if piece is None:
                pieces = ["[UNK]"]
                unknown += 1
                break
            pieces.append(piece)
            start = end
        ids.extend(vocab[piece] for piece in pieces)
    return ids, unknown


def build_title_embeddings(
    project_dir: Path, ratings: pd.DataFrame, model_dir: Path,
) -> tuple[torch.Tensor, dict[str, object]]:
    word_table, model_info = read_word_embedding_table(model_dir)
    vocab = {token: index for index, token in enumerate(model_info.pop("vocab"))}
    unknown_id = vocab["[UNK]"]
    item_info = pd.read_csv(
        project_dir / "data/ml-100k/u.item",
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],
        names=["item_id", "title"],
        engine="python",
    )
    item_order = (
        ratings[["item_id", "item_idx"]]
        .drop_duplicates()
        .sort_values("item_idx")["item_id"]
        .tolist()
    )
    title_by_id = item_info.set_index("item_id")["title"].to_dict()
    vectors = torch.zeros((len(item_order), word_table.shape[1]), dtype=torch.float32)
    token_count = 0
    unknown_count = 0
    for item_idx, item_id in enumerate(item_order):
        if item_id not in title_by_id:
            raise ValueError(f"No title found for MovieLens item_id={item_id}")
        ids, unknown = wordpiece_ids(str(title_by_id[item_id]), vocab, unknown_id)
        unknown_count += unknown
        if ids:
            vectors[item_idx] = word_table[ids].mean(dim=0)
            token_count += len(ids)
        else:
            vectors[item_idx] = word_table[unknown_id]
            unknown_count += 1
    model_info.update({
        "title_count": len(item_order),
        "title_pooling": "arithmetic mean of uncased WordPiece input embeddings; no BERT Transformer layers",
        "token_count": token_count,
        "unknown_basic_token_count": unknown_count,
    })
    return vectors, model_info


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
    parser.add_argument("--output-dir", default="outputs/bert_static_word_embedding_mixed_pool_100")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    project_dir = Path(__file__).resolve().parent
    output_root = project_dir / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = args.bert_model_dir or default_model_dir()
    ratings = load_ratings(project_dir / "data/ml-100k/u.data")
    title_vectors, model_info = build_title_embeddings(project_dir, ratings, model_dir)
    torch.save({"embeddings": title_vectors, "metadata": model_info}, output_root / "title_embeddings.pt")
    train, validation, test = temporal_split(ratings)
    known_items = set(train["item_idx"].astype(int))
    explicit = observed_binary(train)[:3]
    sampling_data = prepare_sampling(train, known_items)
    validation_data = build_ranking_data(validation, train, known_items)
    test_history = pd.concat((train, validation), ignore_index=True)
    test_data = build_ranking_data(test, test_history, known_items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
