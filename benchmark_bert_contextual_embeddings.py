"""Encode complete MovieLens titles with pretrained BERT for MF and NCF ranking.

This is the contextual BERT variant: titles pass through all pretrained BERT
Transformer layers once, then their final-layer token vectors are mean-pooled.
The encoder stays frozen; only the recommendation models and title projections
are trained with the selected sampled-MSE ranking setup.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from benchmark_bert_static_embeddings import (
    MODEL_NAME,
    default_model_dir,
    wordpiece_ids,
)
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


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, positions: int, type_vocab: int, eps: float) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden)
        self.position_embeddings = nn.Embedding(positions, hidden)
        self.token_type_embeddings = nn.Embedding(type_vocab, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_type_ids = torch.zeros_like(input_ids)
        return self.LayerNorm(
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )


class BertSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.heads = heads
        self.head_dim = hidden // heads

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        query = split_heads(self.query(hidden))
        key = split_heads(self.key(hidden))
        value = split_heads(self.value(hidden))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), -10000.0)
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        return context.view(batch, length, width)


class BertSelfOutput(nn.Module):
    def __init__(self, hidden: int, eps: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, context: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(context) + residual)


class BertAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, eps: float) -> None:
        super().__init__()
        self.self_attention = BertSelfAttention(hidden, heads)
        self.output = BertSelfOutput(hidden, eps)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        context = self.self_attention(hidden, attention_mask)
        return self.output(context, hidden)


class BertIntermediate(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden, intermediate)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden), approximate="none")


class BertLayerOutput(nn.Module):
    def __init__(self, intermediate: int, hidden: int, eps: float) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden) + residual)


class BertLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int, eps: float) -> None:
        super().__init__()
        self.attention = BertAttention(hidden, heads, eps)
        self.intermediate = BertIntermediate(hidden, intermediate)
        self.output = BertLayerOutput(intermediate, hidden, eps)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attended = self.attention(hidden, attention_mask)
        return self.output(self.intermediate(attended), attended)


class BertEncoder(nn.Module):
    def __init__(self, layers: int, hidden: int, heads: int, intermediate: int, eps: float) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            BertLayer(hidden, heads, intermediate, eps) for _ in range(layers)
        )


class BertForTitleEncoding(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        eps = float(config.get("layer_norm_eps", 1e-12))
        self.embeddings = BertEmbeddings(
            int(config["vocab_size"]), hidden,
            int(config["max_position_embeddings"]),
            int(config.get("type_vocab_size", 2)), eps,
        )
        self.encoder = BertEncoder(
            int(config["num_hidden_layers"]), hidden,
            int(config["num_attention_heads"]),
            int(config["intermediate_size"]), eps,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden, attention_mask)
        return hidden


def load_safetensors_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Load BERT encoder tensors from safetensors using only torch/numpy."""
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size))
        weights: dict[str, torch.Tensor] = {}
        for key, entry in header.items():
            if key == "__metadata__" or key.startswith("bert.pooler."):
                continue
            if not key.startswith("bert."):
                continue
            if entry["dtype"] != "F32":
                raise TypeError(f"Expected float32 checkpoint tensor for {key}")
            start, end = (int(value) for value in entry["data_offsets"])
            handle.seek(8 + header_size + start)
            raw = handle.read(end - start)
            shape = tuple(int(value) for value in entry["shape"])
            if len(raw) != math.prod(shape) * 4:
                raise ValueError(f"Truncated checkpoint tensor {key}")
            array = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
            target_key = key.removeprefix("bert.")
            target_key = target_key.replace(".attention.self.", ".attention.self_attention.")
            if target_key.endswith(".gamma"):
                target_key = target_key[:-6] + ".weight"
            elif target_key.endswith(".beta"):
                target_key = target_key[:-5] + ".bias"
            weights[target_key] = torch.from_numpy(array)
    config_path = path.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return weights, config


def load_frozen_bert(model_dir: Path, device: torch.device) -> tuple[BertForTitleEncoding, dict[str, object]]:
    weights, config = load_safetensors_state(model_dir / "model.safetensors")
    model = BertForTitleEncoding(config)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    del weights
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config


@torch.inference_mode()
def encode_titles(
    project_dir: Path,
    ratings: pd.DataFrame,
    model_dir: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    model, config = load_frozen_bert(model_dir, device)
    vocab_list = (model_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    vocab = {token: index for index, token in enumerate(vocab_list)}
    item_info = pd.read_csv(
        project_dir / "data/ml-100k/u.item", sep="|", header=None,
        encoding="latin-1", usecols=[0, 1], names=["item_id", "title"], engine="python",
    )
    item_order = (
        ratings[["item_id", "item_idx"]].drop_duplicates().sort_values("item_idx")["item_id"].tolist()
    )
    title_by_id = item_info.set_index("item_id")["title"].to_dict()
    cls_id, sep_id, unk_id = vocab["[CLS]"], vocab["[SEP]"], vocab["[UNK]"]
    title_ids: list[list[int]] = []
    unknown_count = 0
    token_total = 0
    max_positions = int(config["max_position_embeddings"])
    for item_id in item_order:
        if item_id not in title_by_id:
            raise ValueError(f"No title found for MovieLens item_id={item_id}")
        pieces, unknown = wordpiece_ids(str(title_by_id[item_id]), vocab, unk_id)
        unknown_count += unknown
        pieces = pieces[: max_positions - 2]
        token_total += len(pieces)
        title_ids.append([cls_id, *pieces, sep_id])

    output = torch.zeros((len(title_ids), int(config["hidden_size"])), dtype=torch.float32)
    for start in range(0, len(title_ids), batch_size):
        batch_ids = title_ids[start : start + batch_size]
        length = max(map(len, batch_ids))
        input_ids = torch.full((len(batch_ids), length), vocab["[PAD]"], dtype=torch.long, device=device)
        mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(batch_ids):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            mask[row, : len(ids)] = 1
        hidden = model(input_ids, mask)
        # Exclude [CLS], [SEP], and padding; pool contextual final-layer title tokens.
        pooled = torch.stack([
            hidden[row, 1 : len(ids) - 1].mean(dim=0)
            if len(ids) > 2 else hidden[row, 0]
            for row, ids in enumerate(batch_ids)
        ])
        output[start : start + len(batch_ids)] = pooled.cpu()
    model_info = {
        "model": MODEL_NAME,
        "model_snapshot": model_dir.name,
        "hidden_size": int(config["hidden_size"]),
        "layers_run": int(config["num_hidden_layers"]),
        "attention_heads": int(config["num_attention_heads"]),
        "title_count": len(item_order),
        "token_count": token_total,
        "unknown_basic_token_count": unknown_count,
        "pooling": "arithmetic mean of final-layer contextual token vectors, excluding CLS/SEP/PAD",
        "encoder_training": "frozen inference; no fine-tuning",
    }
    return output, model_info


class ContextualTitleMF(nn.Module):
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


class ContextualTitleNCF(nn.Module):
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
) -> dict[str, object]:
    set_seed(config.seed)
    model_type = ContextualTitleMF if model_name == "mf" else ContextualTitleNCF
    model = model_type(n_users, title_vectors, config.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    run_dir = output_root / f"{model_name}_sampled_mse"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pt"
    best_ndcg, best_epoch, stale = -1.0, 0, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        loss, units = train_epoch(
            model, optimizer, "sampled", "mse", train, known_items, explicit,
            sampling_data, epoch, config, device,
        )
        validation = evaluate_ranking(model, validation_data, config.k, device, config.score_batch_size)
        ndcg = float(validation[f"ndcg@{config.k}"])
        history.append({
            "epoch": epoch, "train_loss": loss, "training_units": units,
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
                f"{model_name}/contextual/sample_mse epoch={epoch:03d} "
                f"loss={loss:.4f} val_ndcg={ndcg:.4f} best_epoch={best_epoch}",
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
        "feature": "frozen bert-base-uncased final-layer title token states",
        "fusion": "trainable linear projection to 64 dimensions added to trainable item ID embedding",
        "bert": model_info,
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
    parser.add_argument("--output-dir", default="outputs/bert_contextual_title_mixed_pool_100")
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
