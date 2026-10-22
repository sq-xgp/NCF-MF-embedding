"""Plot validation ranking metrics against epoch for the ten-run benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = (
    ("explicit", "mse", "Observed · MSE", "#147d91", "--"),
    ("explicit", "bce", "Observed · BCE", "#db7a2b", "--"),
    ("sampled", "mse", "Sampled · MSE", "#147d91", "-"),
    ("sampled", "bce", "Sampled · BCE", "#db7a2b", "-"),
    ("sampled", "bpr", "Sampled · BPR", "#7753b8", "-"),
)
METRICS = (
    ("recall@10", "Recall@10"),
    ("hit_ratio@10", "HitRatio@10"),
    ("ndcg@10", "nDCG@10"),
)


def plot(input_dir: Path, output_dir: Path, max_epoch: int | None,
         available_only: bool = False) -> tuple[Path, Path]:
    runs = {}
    for model in ("mf", "ncf"):
        for regime, loss, *_ in RUNS:
            metrics_path = input_dir / f"{model}_{regime}_{loss}" / "metrics.json"
            if available_only and not metrics_path.exists():
                continue
            run = json.loads(metrics_path.read_text(encoding="utf-8"))
            if len(run["history"]) != 1000:
                if available_only:
                    continue
                raise ValueError(f"Expected 1000 epochs: {metrics_path}")
            runs[(model, regime, loss)] = run
    if not runs:
        raise ValueError(f"No completed 1000-epoch runs in {input_dir}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.facecolor": "#f7f6f2",
        "axes.facecolor": "#fffefa",
    })
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
    for row, model in enumerate(("mf", "ncf")):
        for col, (key, title) in enumerate(METRICS):
            ax = axes[row, col]
            for regime, loss, label, color, style in RUNS:
                if (model, regime, loss) not in runs:
                    continue
                run = runs[(model, regime, loss)]
                history = run["history"]
                if max_epoch is not None:
                    history = history[:max_epoch]
                epochs = [int(point["epoch"]) for point in history]
                values = [float(point[f"validation_{key}"]) for point in history]
                ax.plot(
                    epochs, values, color=color, linestyle=style,
                    linewidth=1.25, alpha=0.9, label=label,
                )
                if key == "ndcg@10" and run["best_epoch"] <= (max_epoch or 1000):
                    best = int(run["best_epoch"])
                    ax.scatter([best], [values[best - 1]], color=color, s=25, zorder=4)
            ax.set_title(f"{model.upper()} · {title}", fontweight="bold", pad=10)
            if not any(key[0] == model for key in runs):
                ax.text(0.5, 0.5, "No completed run", transform=ax.transAxes,
                        ha="center", va="center", color="#59635e")
            ax.set_xlim(1, max_epoch or 1000)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel("Validation macro average")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=5, frameon=False)
    range_text = "Epochs 1–1000" if max_epoch is None else f"Epochs 1–{max_epoch}"
    scope = f" · {len(runs)}/10 completed runs" if available_only else ""
    fig.suptitle(f"MF vs NCF · Validation ranking metrics · {range_text}{scope}",
                 fontsize=17, fontweight="bold", y=0.995)
    fig.text(0.5, 0.015,
             "Same MovieLens 100K split and warm-item candidate pool · K=10 · "+
             "Only completed 1000-epoch runs shown · Dots mark best nDCG checkpoints",
             ha="center", fontsize=9, color="#59635e")
    fig.tight_layout(rect=(0.03, 0.04, 0.98, 0.90), h_pad=2.2, w_pad=1.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "1000" if max_epoch is None else str(max_epoch)
    if available_only:
        suffix += "_completed_only"
    png = output_dir / f"mf_ncf_validation_curves_{suffix}.png"
    svg = output_dir / f"mf_ncf_validation_curves_{suffix}.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--available-only", action="store_true",
                        help="Plot only runs with a complete 1000-epoch history")
    args = parser.parse_args()
    for max_epoch in (None, 100):
        png, svg = plot(args.input_dir, args.output_dir, max_epoch,
                        available_only=args.available_only)
        print(png)
        print(svg)


if __name__ == "__main__":
    main()
