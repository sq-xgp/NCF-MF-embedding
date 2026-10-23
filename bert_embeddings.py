"""Frozen BERT word and sentence features using Hugging Face Transformers."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch

MODEL_NAME = "bert-base-uncased"


def default_model_dir() -> Path:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snapshots = hub / f"models--{MODEL_NAME}" / "snapshots"
    available = sorted(snapshots.glob("*/model.safetensors"))
    if not available:
        raise FileNotFoundError(f"Cache {MODEL_NAME} first or pass --bert-model-dir")
    return available[-1].parent


def movie_titles(project_dir: Path, ratings: pd.DataFrame) -> list[str]:
    items = pd.read_csv(
        project_dir / "data/ml-100k/u.item", sep="|", header=None,
        encoding="latin-1", usecols=[0, 1], names=["item_id", "title"],
    ).set_index("item_id")
    order = ratings[["item_id", "item_idx"]].drop_duplicates().sort_values("item_idx")
    titles = items["title"].reindex(order["item_id"])
    if titles.isna().any():
        raise ValueError("Missing movie titles for one or more item IDs")
    return titles.astype(str).tolist()


@torch.no_grad()
def extract_bert_embeddings(
    titles: list[str], model_dir: Path, *, mode: str = "word",
    device: torch.device | str = "cpu", batch_size: int = 32,
) -> tuple[torch.Tensor, dict]:
    """Mean-pool input word embeddings or final-layer contextual token states."""
    # Lazy imports keep the Word2Vec training path independent of Transformers.
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if mode not in {"word", "sentence"} or batch_size <= 0:
        raise ValueError("mode must be word/sentence and batch_size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True, add_pooling_layer=False,
    ).to(device).eval().requires_grad_(False)
    vectors = []
    token_count = unknown_count = 0
    for start in range(0, len(titles), batch_size):
        batch = tokenizer(
            titles[start:start + batch_size], padding=True, truncation=True,
            max_length=model.config.max_position_embeddings,
            return_special_tokens_mask=True, return_tensors="pt",
        ).to(device)
        special = batch.pop("special_tokens_mask").bool()
        mask = batch["attention_mask"].bool() & ~special
        if mode == "word":
            hidden = model.get_input_embeddings()(batch["input_ids"])
            fallback = model.get_input_embeddings().weight[tokenizer.unk_token_id]
        else:
            hidden = model(**batch).last_hidden_state
            fallback = hidden[:, 0]  # [CLS] for an empty title, as in the old encoder.
        counts = mask.sum(dim=1, keepdim=True)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(1)
        vectors.append(torch.where(counts > 0, pooled, fallback).cpu())
        token_count += int(mask.sum())
        unknown_count += int(((batch["input_ids"] == tokenizer.unk_token_id) & mask).sum())
    output = torch.cat(vectors) if vectors else torch.empty((0, model.config.hidden_size))
    pooling = f"arithmetic mean of {'input word embeddings' if mode == 'word' else 'final-layer contextual token vectors'}, excluding CLS/SEP/PAD"
    return output, {
        "model": MODEL_NAME, "model_snapshot": model_dir.name,
        "implementation": "transformers", "transformers_version": transformers.__version__,
        "tokenizer": type(tokenizer).__name__, "feature_mode": mode,
        "embedding_dim": model.config.hidden_size, "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "layers_run": model.config.num_hidden_layers if mode == "sentence" else 0,
        "attention_heads": model.config.num_attention_heads,
        "title_count": len(titles), "token_count": token_count,
        "unknown_token_count": unknown_count, "pooling": pooling, "title_pooling": pooling,
        "encoder_training": "frozen inference; no fine-tuning",
    }
