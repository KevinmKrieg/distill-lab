from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass
class RunSummary:
    label: str
    run_dir: Path
    epochs: int
    best_epoch: int
    best_val_perplexity: float
    best_val_token_accuracy: float
    final_train_loss: float
    parameter_count: int
    checkpoint_size_mb: float


@dataclass
class RunHistory:
    label: str
    run_dir: Path
    rows: List[Dict[str, float]]
    summary: RunSummary


def load_metrics(run_dir: Path) -> List[Dict[str, float]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    rows: List[Dict[str, float]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Metrics file is empty: {metrics_path}")
    return rows


def count_model_parameters(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model_state"]
    return int(sum(tensor.numel() for tensor in model_state.values()))


def summarize_run(run_dir: Path, label: str | None = None) -> RunHistory:
    rows = load_metrics(run_dir)
    best_row = min(rows, key=lambda row: row["val_perplexity"])
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    summary = RunSummary(
        label=label or run_dir.name,
        run_dir=run_dir,
        epochs=len(rows),
        best_epoch=int(best_row["epoch"]),
        best_val_perplexity=float(best_row["val_perplexity"]),
        best_val_token_accuracy=float(best_row["val_token_accuracy"]),
        final_train_loss=float(rows[-1]["train_loss"]),
        parameter_count=count_model_parameters(checkpoint_path),
        checkpoint_size_mb=checkpoint_path.stat().st_size / (1024 * 1024),
    )
    return RunHistory(label=summary.label, run_dir=run_dir, rows=rows, summary=summary)


def write_summary_csv(histories: Sequence[RunHistory], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "run_dir",
                "epochs",
                "best_epoch",
                "best_val_perplexity",
                "best_val_token_accuracy",
                "final_train_loss",
                "parameter_count",
                "checkpoint_size_mb",
            ],
        )
        writer.writeheader()
        for history in histories:
            summary = history.summary
            writer.writerow(
                {
                    "label": summary.label,
                    "run_dir": str(summary.run_dir),
                    "epochs": summary.epochs,
                    "best_epoch": summary.best_epoch,
                    "best_val_perplexity": f"{summary.best_val_perplexity:.4f}",
                    "best_val_token_accuracy": f"{summary.best_val_token_accuracy:.4f}",
                    "final_train_loss": f"{summary.final_train_loss:.4f}",
                    "parameter_count": summary.parameter_count,
                    "checkpoint_size_mb": f"{summary.checkpoint_size_mb:.3f}",
                }
            )


def write_summary_markdown(histories: Sequence[RunHistory], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| run | epochs | best epoch | best val ppl | best val acc | final train loss | params | ckpt mb |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    for history in histories:
        summary = history.summary
        rows.append(
            "| {label} | {epochs} | {best_epoch} | {ppl:.4f} | {acc:.4f} | {loss:.4f} | {params} | {ckpt:.3f} |".format(
                label=summary.label,
                epochs=summary.epochs,
                best_epoch=summary.best_epoch,
                ppl=summary.best_val_perplexity,
                acc=summary.best_val_token_accuracy,
                loss=summary.final_train_loss,
                params=summary.parameter_count,
                ckpt=summary.checkpoint_size_mb,
            )
        )
    output_path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_metric_plot(histories: Sequence[RunHistory], metric_key: str, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 420
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 40, 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    x_max = max(max(int(row["epoch"]) for row in history.rows) for history in histories)
    values = [float(row[metric_key]) for history in histories for row in history.rows]
    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    y_padding = (y_max - y_min) * 0.08
    y_min -= y_padding
    y_max += y_padding

    def x_pos(epoch: float) -> float:
        if x_max <= 1:
            return margin_left
        return margin_left + (epoch - 1) / max(x_max - 1, 1) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + (1 - ((value - y_min) / (y_max - y_min))) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-family="monospace">{_svg_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
    ]

    for tick in range(1, x_max + 1):
        x = x_pos(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 6}" stroke="#444" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{margin_top + plot_height + 22}" text-anchor="middle" font-size="11" font-family="monospace">{tick}</text>')

    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4
        y = y_pos(value)
        parts.append(f'<line x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="#444" stroke-width="1"/>')
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="monospace">{value:.3f}</text>')

    legend_x = margin_left + 8
    legend_y = margin_top + 8
    for index, history in enumerate(histories):
        color = colors[index % len(colors)]
        points = " ".join(
            f"{x_pos(float(row['epoch'])):.2f},{y_pos(float(row[metric_key])):.2f}" for row in history.rows
        )
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        for row in history.rows:
            parts.append(
                f'<circle cx="{x_pos(float(row["epoch"])):.2f}" cy="{y_pos(float(row[metric_key])):.2f}" r="2.8" fill="{color}"/>'
            )
        label_y = legend_y + index * 18
        parts.append(f'<line x1="{legend_x}" y1="{label_y}" x2="{legend_x + 18}" y2="{label_y}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{label_y + 4}" font-size="11" font-family="monospace">{_svg_escape(history.label)}</text>')

    parts.append(f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="12" font-family="monospace">epoch</text>')
    parts.append('</svg>')
    output_path.write_text("\n".join(parts), encoding="utf-8")
