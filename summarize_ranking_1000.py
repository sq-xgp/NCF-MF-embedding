"""Collect ten complete 1000-epoch ranking runs into a compact comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = (
    ("explicit", "mse"), ("explicit", "bce"),
    ("sampled", "mse"), ("sampled", "bce"), ("sampled", "bpr"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model in ("mf", "ncf"):
        for regime, loss in EXPERIMENTS:
            path = args.input_dir / f"{model}_{regime}_{loss}" / "metrics.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            history = run["history"]
            if len(history) != 1000 or history[-1]["epoch"] != 1000:
                raise ValueError(f"Incomplete 1000-epoch run: {path}")
            best = run["best_epoch"]
            final = history[-1]
            test = run["test"]
            rows.append({
                "model": model,
                "negative_source": regime,
                "loss": loss,
                "best_epoch": best,
                "validation_ndcg_best": run["best_validation_ndcg"],
                "validation_ndcg_at_1000": final["validation_ndcg@10"],
                "validation_recall_at_1000": final["validation_recall@10"],
                "validation_hit_ratio_at_1000": final["validation_hit_ratio@10"],
                "test_recall_at_best": test["recall@10"],
                "test_hit_ratio_at_best": test["hit_ratio@10"],
                "test_ndcg_at_best": test["ndcg@10"],
                "evaluated_test_users": test["evaluated_users"],
                "cold_positives_excluded": test["excluded_cold_positive_count"],
            })
    payload = {
        "epochs_trained_per_run": 1000,
        "checkpoint_selection": "best validation nDCG@10 across all 1000 epochs",
        "test_usage": "once after training, using the best validation checkpoint",
        "ranking_metrics_trajectory": "validation only; no per-epoch test metrics",
        "results": rows,
    }
    summary = args.input_dir / "summary_1000.json"
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.input_dir / "summary_1000.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(summary)
    print(csv_path)


if __name__ == "__main__":
    main()
