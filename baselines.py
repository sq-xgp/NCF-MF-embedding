"""Shared entry point for the existing MF/NCF baselines.

Rating models output a normalized score in [0, 1]. Ranking models output
logits; callers choose MSE, BCEWithLogits, or BPR as appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch import nn

from benchmark_ranking_losses import MF as RankingMF
from benchmark_ranking_losses import NCF as RankingNCF
from train import MatrixFactorization, load_ratings, remove_cold_items, temporal_split
from train_ncf import NeuralCollaborativeFiltering


@dataclass(frozen=True)
class ExperimentData:
    ratings: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    rating_validation: pd.DataFrame
    rating_test: pd.DataFrame
    validation_cold_items: int
    test_cold_items: int
    num_users: int
    num_items: int


def load_experiment_data(data_path: str | Path | None = None) -> ExperimentData:
    """Reuse the original ID mapping and per-user chronological 8:1:1 split.

    Raw validation/test are retained for ranking. Rating evaluation uses the
    warm-item subsets, matching the original MF and NCF training scripts.
    """
    path = Path(data_path) if data_path is not None else Path(__file__).resolve().parent / "data/ml-100k/u.data"
    ratings = load_ratings(path)
    train, validation, test = temporal_split(ratings)
    rating_validation, validation_cold_items = remove_cold_items(train, validation)
    rating_test, test_cold_items = remove_cold_items(train, test)
    return ExperimentData(
        ratings=ratings,
        train=train,
        validation=validation,
        test=test,
        rating_validation=rating_validation,
        rating_test=rating_test,
        validation_cold_items=validation_cold_items,
        test_cold_items=test_cold_items,
        num_users=int(ratings["user_idx"].max()) + 1,
        num_items=int(ratings["item_idx"].max()) + 1,
    )


def build_model(
    model_name: str,
    task: str,
    num_users: int,
    num_items: int,
    embedding_dim: int = 64,
    *,
    use_bias: bool = False,
    hidden_dims: tuple[int, ...] | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Construct an unchanged baseline for either rating or ranking."""
    if model_name not in {"mf", "ncf"}:
        raise ValueError("model_name must be 'mf' or 'ncf'")
    if task not in {"rating", "ranking"}:
        raise ValueError("task must be 'rating' or 'ranking'")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if task == "ranking":
        if use_bias or hidden_dims is not None or dropout != 0.0:
            raise ValueError("ranking baselines have fixed architecture; these options only apply to rating")
        return (RankingMF if model_name == "mf" else RankingNCF)(
            num_users, num_items, embedding_dim
        )
    if model_name == "mf":
        if hidden_dims is not None or dropout != 0.0:
            raise ValueError("hidden_dims and dropout only apply to rating NCF")
        return MatrixFactorization(num_users, num_items, embedding_dim, use_bias=use_bias)
    if use_bias:
        raise ValueError("use_bias only applies to rating MF")
    return NeuralCollaborativeFiltering(
        num_users,
        num_items,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims if hidden_dims is not None else (64, 32),
        dropout=dropout,
    )
