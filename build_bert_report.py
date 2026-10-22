"""Build the Chinese MF/NCF BERT and Word2Vec comparison report.

Run `python build_bert_report.py charts`, then run the `pdf` and `verify`
commands with a Python environment containing reportlab and pypdf.
All charts and conclusions use saved experiments; this script never trains.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf"
FIG = OUT / "figures"
DATA = ROOT / "results" / "bert_comparison_report_data.json"
PDF = OUT / "MF_NCF_BERT_Comparison_Report_L2.pdf"
COLORS = {"static": "#128479", "contextual": "#D47A3C", "word2vec": "#7261B5", "word2vec_l2": "#337EBD", "baseline": "#8292A2"}
NAMES = {"static": "BERT 静态", "contextual": "BERT 上下文", "word2vec": "Word2Vec", "word2vec_l2": "Word2Vec L2", "baseline": "ID 基线"}
METRICS = [("recall@10", "Recall@10"), ("hit_ratio@10", "HitRatio@10"), ("ndcg@10", "nDCG@10")]
METHODS = ("static", "contextual", "word2vec", "word2vec_l2")
RUNS = [(model, method) for model in ("mf", "ncf") for method in METHODS]
FOLDERS = {"static": "bert_static_word_embedding_mixed_pool_100",
           "contextual": "bert_contextual_title_mixed_pool_100",
           "word2vec": "word2vec_title_mixed_pool_100",
           "word2vec_l2": "word2vec_title_l2_mixed_pool_100",
           "baseline": "ranking_benchmark_mixed_pool_100"}
CONFIG_KEYS = ("dim", "batch_size", "lr", "weight_decay", "epochs", "patience", "seed", "k")


def records(data, model=None):
    return [rec for family in ("mf", "ncf") if model is None or family == model
            for rec in [data["baselines"][family]] + [c for c in data["cases"] if c["model"] == family]]


def page_count(data):
    return len(data["cases"]) + 10


def collect() -> dict:
    sources = []
    cases = []
    baselines = {}

    def read_source(path):
        raw = path.read_bytes()
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    summaries = {method: read_source(ROOT / "outputs" / folder / "summary.json") for method, folder in FOLDERS.items()}
    reference = summaries["baseline"]
    reference_config = reference["config"]
    coverage = read_source(ROOT / "data" / "pretrained_word2vec" / "coverage.json")
    for model, method in RUNS + [("mf", "baseline"), ("ncf", "baseline")]:
        path = ROOT / "outputs" / FOLDERS[method] / f"{model}_sampled_mse" / "metrics.json"
        run = read_source(path)
        history = run["history"]
        if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
            raise ValueError(f"Empty or non-contiguous epoch history: {path}")
        if (run["model"], run["regime"], run["loss"]) != (model, "sampled", "mse"):
            raise ValueError(f"Unexpected experiment identity: {path}")
        chosen = max(history, key=lambda row: row["validation_ndcg@10"])
        if chosen["epoch"] != run["best_epoch"]:
            raise ValueError(f"Best epoch is inconsistent: {path}")
        for key, value in chosen.items():
            if not math.isclose(value, run["best_validation"][key], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Selected metrics mismatch: {path}, {key}")
        if not math.isclose(run["best_validation_ndcg"], chosen["validation_ndcg@10"], abs_tol=1e-12):
            raise ValueError(f"Selected nDCG mismatch: {path}")
        for row in history:
            if not math.isfinite(row["train_loss"]) or row["train_loss"] < 0:
                raise ValueError(f"Invalid training loss: {path}")
            for key, _ in METRICS:
                if not math.isfinite(row["validation_" + key]) or not 0 <= row["validation_" + key] <= 1:
                    raise ValueError(f"Invalid validation metric: {path}, {key}")
        for key, _ in METRICS:
            if not math.isfinite(run["test"][key]) or not 0 <= run["test"][key] <= 1:
                raise ValueError(f"Invalid test value: {path}, {key}")
        summary = summaries[method]
        matching = [r for r in summary["results"] if (r["model"], r["regime"], r["loss"]) == (model, "sampled", "mse")]
        if len(matching) != 1:
            raise ValueError(f"Missing or duplicate summary entry: {path}")
        for key in ("test", "best_epoch", "best_validation", "best_validation_ndcg"):
            if matching[0][key] != run[key]:
                raise ValueError(f"Run and summary disagree: {path}, {key}")
        run_config = run.get("config", summary["config"])
        for key in CONFIG_KEYS:
            if run_config[key] != reference_config[key] or summary["config"][key] != reference_config[key]:
                raise ValueError(f"Incompatible configuration: {path}, {key}")
        for key in ("split_counts", "validation_population", "test_population"):
            if key in summary and any(summary[key].get(field) != value for field, value in reference[key].items()):
                raise ValueError(f"Incompatible evaluation protocol: {path}, {key}")
        if run["test"]["evaluated_users"] != reference["test_population"]["eligible_users"]:
            raise ValueError(f"Unexpected evaluation population: {path}")
        if run["test"]["excluded_cold_positive_count"] != reference["test_population"]["cold_positives"]:
            raise ValueError(f"Unexpected excluded cold positives: {path}")
        record = {"id": f"{model}_{method}", "model": model, "method": method,
                  "label": f"{model.upper()} + {NAMES[method]}", "epochs_run": len(history),
                  "source": path.relative_to(ROOT).as_posix(), "run": run}
        if method == "baseline":
            baselines[model] = record
        else:
            cases.append(record)
    for case in cases:
        base = baselines[case["model"]]["run"]
        run = case["run"]
        if run["sampling_protocol"] != base["sampling_protocol"]:
            raise ValueError(f"Incompatible negative sampling: {case['id']}")
        if run["test"]["total_candidate_scores"] != base["test"]["total_candidate_scores"]:
            raise ValueError(f"Different evaluation candidate count: {case['id']}")
        if case["method"].startswith("word2vec"):
            projection_input = run["word2vec"]["embedding_dim"]
            if run["word2vec"]["coverage"] != coverage:
                raise ValueError(f"Word2Vec coverage differs from experiment input: {case['id']}")
        else:
            projection_input = run["bert"].get("embedding_dim", run["bert"].get("hidden_size"))
        extra = run["parameter_count_trainable"] - base["parameter_count"]
        if extra != projection_input * reference_config["dim"]:
            raise ValueError(f"Unexpected projection parameter count: {case['id']}")
        case["projection_input_dim"] = projection_input
        case["extra_trainable_parameters"] = extra
    normalization_comparisons = []
    for model in ("mf", "ncf"):
        raw = next(c["run"] for c in cases if (c["model"], c["method"]) == (model, "word2vec"))
        normalized = next(c["run"] for c in cases if (c["model"], c["method"]) == (model, "word2vec_l2"))
        info = normalized["word2vec"]
        if info["normalization_mode"] != "l2" or info["embedding_dim"] != 100:
            raise ValueError("Expected L2 normalization of 100D title vectors")
        for key in ("source_sha256", "title_count", "title_pooling", "coverage_sha256"):
            if info[key] != raw["word2vec"][key]:
                raise ValueError(f"Different Word2Vec inputs in normalization ablation: {key}")
        if normalized["parameter_count_trainable"] != raw["parameter_count_trainable"]:
            raise ValueError("Normalization changed the trainable parameter count")
        for key in ("vector_norm_min", "vector_norm_mean", "vector_norm_max"):
            if not math.isclose(info[key], 1.0, abs_tol=1e-6):
                raise ValueError("Expected unit-norm nonzero title vectors")
        improved = normalized["best_validation_ndcg"] > raw["best_validation_ndcg"]
        normalization_comparisons.append({"model": model, "validation_improved_over_raw": improved,
            "report_inclusion_basis": "validation nDCG@10 improvement over raw Word2Vec; retain all original comparisons"})
        if not improved:
            raise ValueError("L2 report inclusion rule was not met; review the report scope")
    return {"report_date": datetime.now(timezone(timedelta(hours=8))).date().isoformat(), "cases": cases, "baselines": baselines, "sources": sources,
            "baseline_protocol": reference["protocol"], "config": reference_config,
            "split_counts": reference["split_counts"], "validation_population": reference["validation_population"],
            "test_population": reference["test_population"], "word2vec_coverage": coverage,
            "normalization_comparisons": normalization_comparisons}


def load_data():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for source in data["sources"] + data["figure_files"]:
        path = ROOT / source["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Report input changed; regenerate charts first: {path}")
    return data


def charts() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker
    import numpy as np

    data = collect()
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    matplotlib.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                               "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
                               "axes.edgecolor": "#B4C1CC", "text.color": "#223448", "axes.labelcolor": "#223448",
                               "xtick.color": "#536477", "ytick.color": "#536477", "svg.fonttype": "none"})
    FIG.mkdir(parents=True, exist_ok=True)
    specs = [("train_loss", "训练 MSE（越低越好）")] + [("validation_" + key, "验证 " + title + "（越高越好）") for key, title in METRICS]
    limits = {key: (0, max(row[key] for rec in records(data) for row in rec["run"]["history"]) * 1.10)
              for key, _ in specs}
    max_epoch = max(rec["epochs_run"] for rec in records(data))
    data["figure_files"] = []

    def save(fig, name):
        for suffix in ("png", "svg"):
            path = FIG / (name + "." + suffix)
            fig.savefig(path, dpi=230, facecolor="white")
            data["figure_files"].append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)

    for case in data["cases"]:
        fig, axes = plt.subplots(2, 2, figsize=(12.6, 5.25))
        fig.subplots_adjust(left=.072, right=.985, bottom=.105, top=.925, hspace=.60, wspace=.22)
        run = case["run"]
        history = run["history"]
        x = [row["epoch"] for row in history]
        chosen = run["best_epoch"]
        for ax, (key, title) in zip(axes.flat, specs):
            y = [row[key] for row in history]
            ax.plot(x, y, color=COLORS[case["method"]], linewidth=1.9, marker=".", markersize=3.1)
            ax.axvline(chosen, color="#536477", linestyle="--", linewidth=1)
            value = history[chosen - 1][key]
            ax.scatter([chosen], [value], s=43, facecolor="white", edgecolor=COLORS[case["method"]], linewidth=1.8, zorder=5)
            ax.set_title(title, loc="left", pad=10)
            ax.text(.98, 1.04, f"选定值 {value:.4f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
            ax.set_xlim(1, max(2, max(x)))
            ax.set_ylim(*limits[key])
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7, integer=True))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f" if key == "validation_ndcg@10" else "%.2f"))
            ax.grid(axis="y", color="#DDE5EB", linewidth=.6)
        save(fig, case["id"] + "_curves")

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.6))
    fig.subplots_adjust(left=.065, right=.99, top=.89, bottom=.12, hspace=.67, wspace=.28)
    for row, model in enumerate(("mf", "ncf")):
        recs = records(data, model)
        for col, (key, title) in enumerate(METRICS):
            ax = axes[row, col]
            vals = [rec["run"]["test"][key] for rec in recs]
            bars = ax.bar(np.arange(len(recs)), vals, width=.60, color=[COLORS[rec["method"]] for rec in recs])
            ax.bar_label(bars, labels=[f"{value:.4f}" for value in vals], padding=4, fontsize=8)
            ax.set_ylim(0, max(rec["run"]["test"][key] for rec in records(data)) * 1.28)
            ax.set_xticks(np.arange(len(recs)), [NAMES[rec["method"]].replace(" ", "\n") for rec in recs])
            ax.tick_params(axis="x", labelsize=7.8)
            ax.set_title(model.upper() + " / " + title, loc="left", pad=10)
            ax.grid(axis="y", color="#DDE5EB", linewidth=.6)
            ax.set_axisbelow(True)
    save(fig, "test_metrics_comparison")

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.7))
    fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.12, hspace=.67, wspace=.28)
    for row, model in enumerate(("mf", "ncf")):
        for col, (key, title) in enumerate(METRICS):
            ax = axes[row, col]
            fullkey = "validation_" + key
            for rec in records(data, model):
                history = rec["run"]["history"]
                ax.plot([x["epoch"] for x in history], [x[fullkey] for x in history], label=NAMES[rec["method"]],
                        color=COLORS[rec["method"]], linewidth=1.7,
                        linestyle="--" if rec["method"] == "baseline" else "-")
                selected = rec["run"]["best_epoch"]
                ax.scatter([selected], [history[selected - 1][fullkey]], s=28, color=COLORS[rec["method"]], zorder=5)
            ax.set_title(model.upper() + " / 验证 " + title, loc="left", pad=9)
            ax.set_xlim(1, max(2, max_epoch))
            ax.set_ylim(*limits[fullkey])
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
            ax.grid(axis="y", color="#DDE5EB", linewidth=.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(.5, .99), frameon=False, fontsize=10)
    save(fig, "validation_overlay")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(data["cases"]), "history_rows": sum(c["epochs_run"] for c in data["cases"]),
                      "figures": len(data["figure_files"]) // 2, "data": str(DATA)}, ensure_ascii=False))


def pdf() -> None:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle

    data = load_data()
    all_records = records(data)
    cfg = data["config"]
    coverage = data["word2vec_coverage"]
    pdfmetrics.registerFont(TTFont("CN", "C:/Windows/Fonts/msyh.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("CNBold", "C:/Windows/Fonts/msyhbd.ttc", subfontIndex=0))
    pdfmetrics.registerFontFamily("CN", normal="CN", bold="CNBold")
    OUT.mkdir(parents=True, exist_ok=True)
    pagew, pageh = landscape(A4)
    margin = 42
    width = pagew - 2 * margin
    c = canvas.Canvas(str(PDF), pagesize=(pagew, pageh))
    c.setTitle("MF 与 NCF：BERT 与 Word2Vec 嵌入的多指标实验报告")
    c.setAuthor("NCF-MF-embedding experiment report")
    ink, muted = HexColor("#1D3348"), HexColor("#52667A")
    page = 0

    def para(text, x, top, w, size=10.4, color=ink, bold=False, leading=None):
        style = ParagraphStyle("p", fontName="CNBold" if bold else "CN", fontSize=size,
                               leading=leading or size * 1.6, textColor=color, wordWrap="CJK", spaceAfter=0)
        paragraph = Paragraph(text, style)
        _, height = paragraph.wrap(w, pageh)
        if top + height > pageh - 37:
            raise ValueError(f"Text overflows page {page}: {text[:60]}")
        paragraph.drawOn(c, x, pageh - top - height)
        return height

    def start(section, title, subtitle):
        nonlocal page
        if page:
            c.showPage()
        page += 1
        c.setFillColor(HexColor("#0C736B"))
        c.rect(0, pageh - 7, pagew, 7, stroke=0, fill=1)
        para(f"{page:02d} / {section.upper()}", margin, 23, width, size=9.1, color=muted, bold=True)
        para(title, margin, 47, width, size=23, bold=True, leading=31)
        para(subtitle, margin, 89, width, size=10.2, color=muted, leading=15)
        c.setStrokeColor(HexColor("#D8E1E8"))
        c.line(margin, 32, pagew - margin, 32)
        c.setFont("CN", 8)
        c.setFillColor(muted)
        c.drawString(margin, 18, f"MovieLens 100K  |  seed {cfg['seed']}  |  报告生成：{data['report_date']}")
        c.drawRightString(pagew - margin, 18, f"{page:02d} / {page_count(data):02d}")

    def table(rows, top, widths, size=9.5, highlights=None):
        cellstyle = ParagraphStyle("cell", fontName="CN", fontSize=size, leading=size * 1.5, wordWrap="CJK", textColor=ink)
        headstyle = ParagraphStyle("head", parent=cellstyle, fontName="CNBold", textColor=white)
        cooked = [[Paragraph(str(cell), headstyle if i == 0 else cellstyle) for cell in row] for i, row in enumerate(rows)]
        result = Table(cooked, colWidths=widths, repeatRows=1, hAlign="LEFT")
        commands = [("BACKGROUND", (0, 0), (-1, 0), HexColor("#223F55")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]
        for i in range(1, len(rows)):
            commands += [("BACKGROUND", (0, i), (-1, i), HexColor("#F0F5F7") if i % 2 else white),
                         ("LINEBELOW", (0, i), (-1, i), .4, HexColor("#DCE5EB"))]
        for cell in highlights or []:
            commands.append(("BACKGROUND", cell, cell, HexColor("#DDF1EB")))
        result.setStyle(TableStyle(commands))
        _, height = result.wrap(sum(widths), pageh)
        if top + height > pageh - 40:
            raise ValueError(f"Table overflow on page {page}")
        result.drawOn(c, margin, pageh - top - height)
        return height

    def highlights(split, start_column, recs):
        cells = []
        for offset, (key, _) in enumerate(METRICS):
            metric = key if split == "test" else "validation_" + key
            best = max(rec["run"][split][metric] for rec in recs)
            cells += [(start_column + offset, i) for i, rec in enumerate(recs, start=1)
                      if math.isclose(rec["run"][split][metric], best, rel_tol=0, abs_tol=1e-12)]
        return cells

    def fig(name, top, height):
        c.drawImage(str(FIG / (name + ".png")), margin, pageh - top - height, width=width, height=height,
                    preserveAspectRatio=True, anchor="c", mask="auto")

    def section(text, top):
        para(text, margin, top, width, size=13, bold=True)

    for model in ("mf", "ncf"):
        recs = records(data, model)
        start("final test results", model.upper() + "：五种表示的最终测试对比", "综合报告包含八组标题特征模型与两个 ID 基线；Word2Vec L2 为新增归一化对照。")
        rows = [["实验情况", "最佳<br/>epoch", "实际<br/>轮数", "Recall@10", "HitRatio@10", "nDCG@10", "nDCG 相对<br/>ID 基线"]]
        for rec in recs:
            run, base = rec["run"], data["baselines"][model]["run"]
            relative = 100 * (run["test"]["ndcg@10"] / base["test"]["ndcg@10"] - 1)
            rows.append([rec["label"], run["best_epoch"], rec["epochs_run"], *[f"{run['test'][key]:.4f}" for key, _ in METRICS],
                         "参照" if rec["method"] == "baseline" else f"{relative:+.2f}%"])
        height = table(rows, 130, [182, 60, 62, 100, 110, 100, width - 614], highlights=highlights("test", 3, recs))
        top = 144 + height
        top += para("绿色单元格表示该项测试指标的最高原始值，并列值均高亮。每行三个测试指标来自同一验证选定 checkpoint；高亮仅描述测试结果，不作为模型选择依据。", margin, top, width, size=9.8, color=muted) + 14
        top += para("nDCG 相对变化 =（该方法 / 同模型 ID 基线 - 1）× 100%；表格保留四位小数，相对变化使用未舍入数值。", margin, top, width, size=10) + 24
        normalized = next(r["run"] for r in recs if r["method"] == "word2vec_l2")
        raw = next(r["run"] for r in recs if r["method"] == "word2vec")
        rel_raw = 100 * (normalized["test"]["ndcg@10"] / raw["test"]["ndcg@10"] - 1)
        rel_id = 100 * (normalized["test"]["ndcg@10"] / base["test"]["ndcg@10"] - 1)
        para(f"<b>新增 L2 对照：</b>测试 nDCG 相对原始 Word2Vec {rel_raw:+.2f}%，相对 ID 基线 {rel_id:+.2f}%。原始版本保留，便于区分归一化收益与相对基线的收益。", margin, top, width, size=10.6)

    for model in ("mf", "ncf"):
        recs = records(data, model)
        start("selected validation results", model.upper() + "：验证指标与选定训练误差", "checkpoint 以最高验证 nDCG@10 选取，原始值相同则取最早 epoch；训练 MSE 对应同一轮。")
        rows = [["实验情况", "选定训练 MSE", "验证 Recall@10", "验证 HitRatio@10", "验证 nDCG@10"]]
        for rec in recs:
            chosen = rec["run"]["best_validation"]
            rows.append([rec["label"], f"{chosen['train_loss']:.4f}", *[f"{chosen['validation_' + key]:.4f}" for key, _ in METRICS]])
        height = table(rows, 130, [190, 135, 142, 147, width - 614], highlights=highlights("best_validation", 2, recs))
        top = 146 + height
        top += para("绿色为各项验证排序指标的最高值；不同指标的最高值可能来自不同方法。训练 MSE 越低越好，选模型的标准始终是验证 nDCG。", margin, top, width, size=9.8, color=muted) + 17
        best = max(recs, key=lambda rec: rec["run"]["best_validation"]["validation_ndcg@10"])
        top += para(f"按验证 nDCG 选择：<b>{NAMES[best['method']]}</b>（{best['run']['best_validation']['validation_ndcg@10']:.4f}）。", margin, top, width, size=10.6) + 17
        raw = next(r["run"] for r in recs if r["method"] == "word2vec")
        normalized = next(r["run"] for r in recs if r["method"] == "word2vec_l2")
        para(f"<b>L2 纳入依据：</b>验证 nDCG 从原始 Word2Vec 的 {raw['best_validation_ndcg']:.4f} 提升到 {normalized['best_validation_ndcg']:.4f}，满足训练前说明的报告纳入标准。", margin, top, width, size=10.5)

    start("protocol and feature fusion", "统一训练口径与四种标题特征", "Word2Vec 原始版与 L2 版均不调用 BERT；冻结标题特征经可训练投影后与电影 ID 向量相加。")
    splits = data["split_counts"]
    left = (f"<b>训练配置：</b>混合负样本池 + MSE，维度 {cfg['dim']}，Adam 学习率 {cfg['lr']}，weight decay {cfg['weight_decay']:g}，batch {cfg['batch_size']}；最多 {cfg['epochs']} 轮，patience {cfg['patience']}，seed {cfg['seed']}。<br/>"
            f"<b>数据划分：</b>按用户时间顺序 8:1:1；训练 {splits['train']:,}、验证 {splits['validation']:,}、测试 {splits['test']:,} 条。验证评估 {data['validation_population']['eligible_users']} 位用户，测试评估 {data['test_population']['eligible_users']} 位用户。")
    right = ("<b>Recall@10：</b>每位用户的相关电影被 Top-10 找回的比例，再对用户取平均。<br/>"
             "<b>HitRatio@10：</b>Top-10 至少命中一部相关电影的用户比例。<br/>"
             "<b>nDCG@10：</b>命中位置越靠前，折扣后的奖励越大，再归一化。<br/><b>训练 MSE：</b>二值采样标签上的均方误差，越低越好。")
    para(left, margin, 128, width / 2 - 15, size=9.8)
    para(right, margin + width / 2 + 12, 128, width / 2 - 12, size=9.8)
    rows = [["标题特征", "冻结特征的生成方式", "可训练投影 / 新增参数"]]
    for method, description in [("static", "BERT 输入词向量的 WordPiece 均值，不运行编码器"),
                                ("contextual", "BERT 最后一层标题 token 均值，排除特殊符号和填充"),
                                ("word2vec", "标题中词表内 center 词向量取均值；按电影 ID 对齐，不额外归一化"),
                                ("word2vec_l2", "对上述 100 维电影均值向量逐行 L2 归一化，再做可训练线性投影")]:
        case = next(c for c in data["cases"] if c["method"] == method)
        rows.append([NAMES[method], description, f"{case['projection_input_dim']} → {cfg['dim']}，无偏置<br/>+{case['extra_trainable_parameters']:,} 参数"])
    height = table(rows, 245, [132, 405, width - 537], size=9.1)
    top = 257 + height
    top += para("<b>标签与负采样：</b>4-5 分为正反馈，1-2 分为显式负反馈，3 分不提供训练标签。每轮为每个训练正样本配一个负样本，从该用户训练期显式负反馈与训练期未评分、全局已见电影的并集中均匀抽取。", margin, top, width, size=9.1, leading=14) + 8
    para("<b>候选与评估：</b>候选为训练期全局已知电影，并排除用户先前交互；测试先前交互含训练与验证。各模型在选定 checkpoint 上计算一次最终测试结果。", margin, top, width, size=9.1, leading=14)

    start("test comparison", "三个测试指标：五种表示方法并列", "灰：ID 基线；绿：BERT 静态；橙：BERT 上下文；紫：原始 Word2Vec；蓝：Word2Vec L2。")
    fig("test_metrics_comparison", 115, 355)
    para("每根柱子代表一次最终测试评估。纵轴从 0 开始，同一指标在 MF 与 NCF 两行采用相同量程；表格中的相对变化与图中的绝对分数不要混淆。", margin, 487, width, size=10.2)

    for case in data["cases"]:
        run = case["run"]
        start("training curves", case["label"] + "：四项逐轮指标", f"选定 epoch {run['best_epoch']} / 实际训练 {case['epochs_run']} 轮；虚线与空心圆标记同一个验证 nDCG 最优 checkpoint。")
        fig(case["id"] + "_curves", 120, 321)
        vals = run["test"]
        para(f"<b>最终测试：</b>Recall@10 = {vals['recall@10']:.4f}　｜　HitRatio@10 = {vals['hit_ratio@10']:.4f}　｜　nDCG@10 = {vals['ndcg@10']:.4f}", margin, 451, width, size=10.6)
        note = (f"第 {run['best_epoch']} 轮的验证 nDCG 为 {run['best_validation']['validation_ndcg@10']:.4f}，选定训练 MSE 为 {run['best_validation']['train_loss']:.4f}。"
                f"记录截至第 {case['epochs_run']} 轮；后续训练损失变化不改变已选 checkpoint。")
        para(note, margin, 481, width, size=10.2)
        para("曲线为逐轮原始值，不平滑、不延长结束后的序列。八组实验页面对同一指标使用相同纵轴量程。", margin, 523, width, size=8.8, color=muted)

    max_epoch = max(rec["epochs_run"] for rec in all_records)
    start("learning trajectory", "验证曲线叠加：每个模型的五种表示", f"完整历史按实际训练轮数截断；圆点为各自选定 checkpoint，横轴统一到第 {max_epoch} 轮。")
    fig("validation_overlay", 110, 365)
    para("同一模型家族共享训练口径与评价人群，曲线展示真实波动。曲线长度表示训练轮数；日志未提供所有方法可比的耗时记录，因此不据此比较运行速度。", margin, 489, width, size=10.2)

    start("normalization ablation", "Word2Vec：归一化前后的直接对照", "唯一输入改动为逐电影标题向量的 L2 归一化；两组均按验证 nDCG 提升纳入报告。")
    rows = [["模型与标题表示", "验证 nDCG@10", "测试 Recall@10", "测试 HitRatio@10", "测试 nDCG@10"]]
    for model in ("mf", "ncf"):
        for method in ("word2vec", "word2vec_l2"):
            rec = next(r for r in records(data, model) if r["method"] == method)
            run = rec["run"]
            rows.append([rec["label"], f"{run['best_validation_ndcg']:.4f}", *[f"{run['test'][key]:.4f}" for key, _ in METRICS]])
    height = table(rows, 128, [190, 137, 142, 147, width - 616], size=10)
    top = 144 + height
    rows = [["L2 版本", "验证 nDCG 相对<br/>原始 Word2Vec", "测试 nDCG 相对<br/>原始 Word2Vec", "测试 nDCG 相对<br/>ID 基线"]]
    statements = []
    for model in ("mf", "ncf"):
        recs = {r["method"]: r["run"] for r in records(data, model)}
        raw, norm, base = recs["word2vec"], recs["word2vec_l2"], recs["baseline"]
        val_change = 100 * (norm["best_validation_ndcg"] / raw["best_validation_ndcg"] - 1)
        test_change = 100 * (norm["test"]["ndcg@10"] / raw["test"]["ndcg@10"] - 1)
        id_change = 100 * (norm["test"]["ndcg@10"] / base["test"]["ndcg@10"] - 1)
        rows.append([model.upper(), f"{val_change:+.2f}%", f"{test_change:+.2f}%", f"{id_change:+.2f}%"])
        verdict = "高于" if id_change > 0 else "低于" if id_change < 0 else "等于"
        statements.append(f"<b>{model.upper()}：</b>L2 测试 nDCG {verdict} ID 基线；相对变化 {id_change:+.2f}%。")
    top += table(rows, top, [130, 210, 210, width - 550], size=9.8) + 14
    top += para("<br/>".join(statements), margin, top, width, size=10.2, leading=16) + 12
    top += para("<b>处理位置：</b>100 维电影标题均值向量 / max(L2 长度, 1e-12) → 可训练 100→64 投影 → 与电影 ID 相加。1,682 个非零向量长度由均值 26.5576 变为约 1；源向量文件保持原样。", margin, top, width, size=9.4, leading=15) + 10
    para("原始与 L2 的参数初值、负采样及训练配置一致；按同一早停规则得到不同训练轮数。本页比较完整训练流程，单 seed 结果不构成统计显著性证据。", margin, top, width, size=9.2, color=muted, leading=14)

    start("observations", "Word2Vec 的实际变化与方法选择", "所有差值由保存的原始指标计算；以下结论仅适用于当前数据、配置和单次随机种子。")
    top = 130
    for model in ("mf", "ncf"):
        recs = records(data, model)
        best_val = max(recs, key=lambda rec: rec["run"]["best_validation"]["validation_ndcg@10"])
        best_test = max(recs, key=lambda rec: rec["run"]["test"]["ndcg@10"])
        text = (f"<b>{model.upper()}：</b>按验证 nDCG 选择的方法为 {NAMES[best_val['method']]}"
                f"（验证 {best_val['run']['best_validation']['validation_ndcg@10']:.4f}，其测试 {best_val['run']['test']['ndcg@10']:.4f}）。"
                f"测试 nDCG 的最高观测值来自 {NAMES[best_test['method']]}（{best_test['run']['test']['ndcg@10']:.4f}），该观察不作为选模型依据。")
        top += para(text, margin, top, width, size=10.4) + 15
    section("未归一化 Word2Vec 的历史对照：相对 ID 基线", top)
    top += 31
    rows = [["模型", "Δ Recall@10<br/>百分点", "Δ HitRatio@10<br/>百分点", "Δ nDCG@10<br/>百分点", "nDCG 相对变化"]]
    for model in ("mf", "ncf"):
        word = next(c for c in data["cases"] if c["model"] == model and c["method"] == "word2vec")["run"]["test"]
        base = data["baselines"][model]["run"]["test"]
        rows.append([model.upper(), *[f"{100 * (word[key] - base[key]):+.3f}" for key, _ in METRICS],
                     f"{100 * (word['ndcg@10'] / base['ndcg@10'] - 1):+.2f}%"])
    top += table(rows, top, [120, 158, 158, 158, width - 594], size=10) + 15
    comparisons = []
    for model in ("mf", "ncf"):
        recs = {rec["method"]: rec for rec in records(data, model)}
        word = recs["word2vec"]["run"]["test"]["ndcg@10"]
        deltas = "；".join(f"相对 {NAMES[method]} {100 * (word - recs[method]['run']['test']['ndcg@10']):+.3f} 个百分点" for method in ("static", "contextual"))
        comparisons.append(f"<b>{model.upper()} Word2Vec 测试 nDCG：</b>{deltas}。")
    top += para("<br/>".join(comparisons), margin, top, width, size=10.1) + 12
    para("百分点表示分数之差乘以 100；相对变化表示比值变化。仅凭本次单 seed 的差异，不能断言某种词向量或编码方式普遍更优。", margin, top, width, size=9.4, color=muted)

    start("limits and provenance", "覆盖率、比较限制与可复现来源", "完整保留 BERT、原始 Word2Vec 和 L2 Word2Vec；没有生成逐轮测试曲线。")
    section("解释结果时需要保留的边界", 125)
    limitations = (f"1. Word2Vec 元数据记录维度 {coverage['embedding_dim']}、词表 {coverage['vocabulary_size']:,}；"
                   f"电影覆盖 {coverage['covered_movies']:,}/{coverage['movie_count']:,}（{coverage['movie_coverage']:.2%}），"
                   f"标题词命中 {coverage['matched_title_tokens']:,}/{coverage['total_title_tokens']:,}（{coverage['token_coverage']:.2%}）。"
                   "电影均有向量，不等于每个词均在词表内。外部预训练语料尚未核实；仅能确认本次推荐训练不读取验证/测试标签，不能据此断言整个预训练链路没有泄漏。<br/>"
                   "2. 所有结果只有一个 seed，未给出置信区间或显著性检验。原始与 L2 Word2Vec 初始可训练参数一致，各增加 6,400 个投影参数；BERT 增加 49,152 个。与 BERT/ID 的比较存在参数量、初值及预训练来源差异。<br/>"
                   f"3. 评价仅覆盖训练期全局已知电影；测试排除了 {data['test_population']['cold_positives']} 条冷启动正反馈，不衡量冷启动收益。BERT 上下文编码使用项目内 PyTorch 实现，未新增与标准库的数值一致性验证。<br/>"
                   "4. L2 对照在观察历史实验后提出，按验证 nDCG 选 checkpoint 并判断纳入，未按本轮测试结果继续调参。原始与 L2 使用相同早停规则，训练轮数不同；稳健性需要多 seed 或独立数据确认。<br/>"
                   "5. 记录包含训练 MSE 与验证排序指标，不包含可比的验证损失、逐轮测试指标及全套运行时间。")
    top = 151 + para(limitations, margin, 151, width, size=9.5, leading=15.5)
    section("原始记录与报告材料", top + 12)
    source_lines = [f"{NAMES[method]}：outputs/{FOLDERS[method]}/" for method in ("baseline",) + METHODS]
    top += 40 + para("<br/>".join(source_lines), margin, top + 40, width, size=9, leading=14)
    detail = ("各目录内 mf_sampled_mse/metrics.json、ncf_sampled_mse/metrics.json 提供逐轮记录，summary.json 用于交叉核对。<br/>"
              "Word2Vec 覆盖率：data/pretrained_word2vec/coverage.json；模型输入采用该目录提供的电影标题向量。<br/>"
              "数据快照：results/bert_comparison_report_data.json，含八组与基线原始历史、配置、覆盖率、来源及图表 SHA-256。<br/>"
              "生成脚本：build_bert_report.py（charts / pdf / verify）；图表：output/pdf/figures/（PNG 与 SVG）。")
    para(detail, margin, top + 9, width, size=8.7, leading=13.5)
    if page != page_count(data):
        raise ValueError(f"Report page accounting mismatch: {page}")
    c.save()
    print(str(PDF))


def verify() -> None:
    from pypdf import PdfReader
    data = load_data()
    reader = PdfReader(PDF)
    if len(reader.pages) != page_count(data):
        raise ValueError(f"Unexpected page count: {len(reader.pages)}")
    texts = [page.extract_text() for page in reader.pages]
    checked = 0
    for rec in records(data):
        test_page = 0 if rec["model"] == "mf" else 1
        validation_page = test_page + 2
        if rec["label"] not in texts[test_page] or rec["label"] not in texts[validation_page]:
            raise ValueError(f"Missing comparison row: {rec['label']}")
        for key, _ in METRICS:
            for page_index, expected in [(test_page, rec["run"]["test"][key]), (validation_page, rec["run"]["best_validation"]["validation_" + key])]:
                if f"{expected:.4f}" not in texts[page_index]:
                    raise ValueError(f"Missing comparison value {expected:.4f} on page {page_index + 1}")
                checked += 1
        if f"{rec['run']['best_validation']['train_loss']:.4f}" not in texts[validation_page]:
            raise ValueError(f"Missing selected loss: {rec['id']}")
        checked += 1
    for index, case in enumerate(data["cases"], start=6):
        if case["label"] not in texts[index]:
            raise ValueError(f"Wrong curve page: {index + 1}")
        for key, _ in METRICS:
            if f"{case['run']['test'][key]:.4f}" not in texts[index]:
                raise ValueError(f"Wrong curve-page test value: {case['id']}, {key}")
    ablation_text = texts[7 + len(data["cases"])]
    for model in ("mf", "ncf"):
        recs = {r["method"]: r["run"] for r in records(data, model)}
        raw, norm, base = recs["word2vec"], recs["word2vec_l2"], recs["baseline"]
        for method in ("word2vec", "word2vec_l2"):
            run = recs[method]
            for value in [run["best_validation_ndcg"]] + [run["test"][key] for key, _ in METRICS]:
                if f"{value:.4f}" not in ablation_text:
                    raise ValueError("Missing normalization comparison value")
                checked += 1
        changes = [100 * (norm["best_validation_ndcg"] / raw["best_validation_ndcg"] - 1),
                   100 * (norm["test"]["ndcg@10"] / raw["test"]["ndcg@10"] - 1),
                   100 * (norm["test"]["ndcg@10"] / base["test"]["ndcg@10"] - 1)]
        for value in changes:
            if f"{value:+.2f}%" not in ablation_text:
                raise ValueError("Missing normalization percentage change")
            checked += 1
    for index, text in enumerate(texts, start=1):
        if f"{index:02d} / {page_count(data):02d}" not in text:
            raise ValueError(f"Missing page number: {index}")
    print(json.dumps({"pages": len(reader.pages), "bytes": PDF.stat().st_size,
                      "table_values_checked": checked, "history_rows": sum(c['epochs_run'] for c in data['cases']),
                      "sha256": hashlib.sha256(PDF.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["charts", "pdf", "verify"])
    operation = parser.parse_args().operation
    {"charts": charts, "pdf": pdf, "verify": verify}[operation]()
