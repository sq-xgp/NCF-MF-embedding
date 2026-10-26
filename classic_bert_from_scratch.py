"""Handwritten BERT encoder and MLM training, with randomly initialized weights.

Only the WordPiece tokenizer comes from Transformers. Default: a small learning
configuration; --size base selects 12 layers / 768 hidden units / 12 heads.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer


@dataclass
class BertConfig:
    vocab_size: int
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 512
    max_positions: int = 512
    pad_token_id: int = 0
    dropout: float = 0.1


class BertEmbeddings(nn.Module):
    """1. token + position + segment embeddings, then LayerNorm and Dropout."""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position = nn.Embedding(config.max_positions, config.hidden_size)
        self.segment = nn.Embedding(2, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, token_type_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.word(input_ids) + self.position(positions) + self.segment(token_type_ids)
        return self.dropout(self.norm(hidden))


class MultiHeadSelfAttention(nn.Module):
    """2. Bidirectional multi-head attention: softmax(QK^T / sqrt(d)) V."""
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, attention_mask):
        batch, length, width = hidden.shape
        query = self.query(hidden).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        # Only padding keys are hidden; there is no causal/future-token mask.
        scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        probabilities = self.dropout(torch.softmax(scores, dim=-1))
        context = (probabilities @ value).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(context)


class BertLayer(nn.Module):
    """3. Attention + residual + LayerNorm; FFN + residual + LayerNorm."""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size), nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, attention_mask):
        hidden = self.attention_norm(hidden + self.dropout(self.attention(hidden, attention_mask)))
        return self.output_norm(hidden + self.dropout(self.ffn(hidden)))


class ClassicBert(nn.Module):
    """4. Stack encoder layers and return token states [B, L, hidden_size]."""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_layers)])

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return hidden


class ClassicBertForMLM(nn.Module):
    """5. Predict masked words; the decoder shares the input word table."""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = ClassicBert(config)
        self.mlm_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=1e-12),
        )
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
            elif module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.bert(input_ids, attention_mask, token_type_ids)
        hidden = self.mlm_transform(hidden)
        return F.linear(hidden, self.bert.embeddings.word.weight, self.mlm_bias)


def mask_tokens(batch, tokenizer):
    """Select 15% of content tokens: 80% MASK, 10% random, 10% unchanged."""
    input_ids, attention_mask, token_type_ids, special_mask = batch
    valid = attention_mask.bool() & ~special_mask.bool()
    selected = (torch.rand(input_ids.shape) < 0.15) & valid
    if not selected.any():  # Short batches must still have a supervised target.
        positions = valid.nonzero()
        if len(positions) == 0:
            raise ValueError("No content tokens available for MLM")
        row, col = positions[torch.randint(len(positions), ())]
        selected[row, col] = True
    labels = input_ids.clone()
    labels[~selected] = -100
    corrupted = input_ids.clone()
    choices = torch.rand(input_ids.shape)
    corrupted[selected & (choices < 0.8)] = tokenizer.mask_token_id
    random_positions = selected & (choices >= 0.8) & (choices < 0.9)
    random_words = torch.randint(len(tokenizer), input_ids.shape)
    corrupted[random_positions] = random_words[random_positions]
    return {"input_ids": corrupted, "attention_mask": attention_mask, "token_type_ids": token_type_ids}, labels


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    total_loss = correct = total = 0
    for inputs, labels in batches:
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)
        logits = model(**inputs)
        selected = labels != -100
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        count = int(selected.sum())
        total_loss += float(loss) * count
        correct += int((logits.argmax(-1)[selected] == labels[selected]).sum())
        total += count
    return {"loss": total_loss / total, "masked_accuracy": correct / total, "masked_tokens": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or not 3 <= args.max_length <= 512 or args.lr <= 0:
        raise ValueError("Positive epochs/batch-size/lr and max-length in [3, 512] are required")
    root = Path(__file__).resolve().parent
    output = args.output_dir or root / "outputs" / f"classic_bert_scratch_{args.size}"
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Movie titles are the corpus; no rating/recommendation labels are used.
    with (root / "data/ml-100k/u.item").open(encoding="latin-1", newline="") as file:
        records = sorted(csv.reader(file, delimiter="|", quoting=csv.QUOTE_NONE), key=lambda row: int(row[0]))
    titles = [row[1] for row in records]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoded = tokenizer(titles, padding="max_length", truncation=True, max_length=args.max_length,
                        return_special_tokens_mask=True, return_tensors="pt")
    dataset = TensorDataset(*(encoded[key] for key in ("input_ids", "attention_mask", "token_type_ids", "special_tokens_mask")))
    val_count = max(1, len(dataset) // 10)
    train_data, val_data = random_split(dataset, [len(dataset) - val_count, val_count], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 1)
        val_batches = [mask_tokens(batch, tokenizer) for batch in val_loader]

    config = BertConfig(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)
    if args.size == "base":
        config.hidden_size, config.num_layers, config.num_heads, config.intermediate_size = 768, 12, 12, 3072
    model = ClassicBertForMLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    initial_bias = model.mlm_bias.detach().cpu().clone()
    print(json.dumps({"device": str(device), "config": asdict(config), "trainable_parameters": sum(p.numel() for p in model.parameters()),
                      "train_titles": len(train_data), "validation_titles": len(val_data)}), flush=True)
    before = evaluate(model, val_batches, device)
    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_targets = 0
        for batch in train_loader:
            inputs, labels = mask_tokens(batch, tokenizer)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int((labels != -100).sum())
            total_loss += float(loss.detach()) * count
            total_targets += count
        validation = evaluate(model, val_batches, device)
        row = {"epoch": epoch, "train_loss": total_loss / total_targets,
               "validation_loss": validation["loss"], "validation_masked_accuracy": validation["masked_accuracy"]}
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save({"config": asdict(config), "model_state_dict": model.state_dict(), "epoch": epoch}, output / "best_model.pt")
        print(json.dumps(row), flush=True)
    tokenizer.save_pretrained(output / "tokenizer")
    summary = {"initialization": "random; no pretrained model weights loaded", "task": "MLM only",
               "size": args.size, "config": asdict(config), "seed": args.seed,
               "train_titles": len(train_data), "validation_titles": len(val_data),
               "validation_masking": "fixed once; training masks resampled each batch",
               "before_training": before, "history": history, "best_validation_loss": best_loss,
               "parameters_updated": not torch.equal(initial_bias, model.mlm_bias.detach().cpu())}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {output / 'best_model.pt'}", flush=True)


if __name__ == "__main__":
    main()
