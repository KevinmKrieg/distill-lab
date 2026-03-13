#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reporting import summarize_run, write_metric_plot, write_summary_csv, write_summary_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize training runs and generate comparison plots.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories containing metrics.jsonl and best.pt.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels aligned with run_dirs.")
    parser.add_argument("--output-dir", default="results/reports/latest", help="Directory for generated summary artifacts.")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) not in {0, len(args.run_dirs)}:
        raise ValueError("--labels must be omitted or match the number of run directories.")

    labels = args.labels if args.labels else [None] * len(args.run_dirs)
    histories = [summarize_run(Path(run_dir), label) for run_dir, label in zip(args.run_dirs, labels)]
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    write_summary_csv(histories, output_dir / "summary.csv")
    write_summary_markdown(histories, output_dir / "summary.md")
    write_metric_plot(histories, "val_perplexity", "Validation Perplexity", plots_dir / "val_perplexity.svg")
    write_metric_plot(histories, "val_token_accuracy", "Validation Token Accuracy", plots_dir / "val_token_accuracy.svg")
    write_metric_plot(histories, "train_loss", "Train Loss", plots_dir / "train_loss.svg")

    print(f"wrote {output_dir / 'summary.csv'}")
    print(f"wrote {output_dir / 'summary.md'}")
    print(f"wrote {plots_dir / 'val_perplexity.svg'}")
    print(f"wrote {plots_dir / 'val_token_accuracy.svg'}")
    print(f"wrote {plots_dir / 'train_loss.svg'}")


if __name__ == "__main__":
    main()
