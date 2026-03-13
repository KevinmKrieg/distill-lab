#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking import benchmark_checkpoint, save_benchmark
from src.reporting import summarize_run, write_summary_csv, write_summary_markdown
from src.utils import pick_device


def resolve_config_path(run_dir: Path, explicit_config: str | None) -> Path:
    if explicit_config is not None:
        return Path(explicit_config)
    candidate = run_dir / 'run_config.yaml'
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f'Missing run config for {run_dir}. Pass --configs or train with the updated trainer.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark saved run directories for latency, throughput, and memory.')
    parser.add_argument('run_dirs', nargs='+', help='Run directories containing best.pt.')
    parser.add_argument('--labels', nargs='*', default=None, help='Optional labels aligned with run_dirs.')
    parser.add_argument('--configs', nargs='*', default=None, help='Optional config paths aligned with run_dirs.')
    parser.add_argument('--output-dir', default='results/reports/benchmarks', help='Directory for comparison summaries.')
    parser.add_argument('--warmup-batches', type=int, default=5, help='Number of warmup batches before timing.')
    parser.add_argument('--benchmark-batches', type=int, default=20, help='Number of timed batches.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to use for benchmarking.')
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) not in {0, len(args.run_dirs)}:
        raise ValueError('--labels must be omitted or match the number of run directories.')
    if args.configs is not None and len(args.configs) not in {0, len(args.run_dirs)}:
        raise ValueError('--configs must be omitted or match the number of run directories.')

    labels = args.labels if args.labels else [None] * len(args.run_dirs)
    configs = args.configs if args.configs else [None] * len(args.run_dirs)
    device = pick_device() if args.device == 'auto' else __import__('torch').device(args.device)

    for run_dir_str, label, config_path in zip(args.run_dirs, labels, configs):
        run_dir = Path(run_dir_str)
        checkpoint_path = run_dir / 'best.pt'
        resolved_config = resolve_config_path(run_dir, config_path)
        result = benchmark_checkpoint(
            config_path=resolved_config,
            checkpoint_path=checkpoint_path,
            label=label or run_dir.name,
            device=device,
            warmup_batches=args.warmup_batches,
            benchmark_batches=args.benchmark_batches,
        )
        save_benchmark(result, run_dir / 'benchmark.json')
        print(f'wrote {run_dir / "benchmark.json"}')

    histories = [summarize_run(Path(run_dir), label) for run_dir, label in zip(args.run_dirs, labels)]
    output_dir = Path(args.output_dir)
    write_summary_csv(histories, output_dir / 'summary.csv')
    write_summary_markdown(histories, output_dir / 'summary.md')
    print(f'wrote {output_dir / "summary.csv"}')
    print(f'wrote {output_dir / "summary.md"}')


if __name__ == '__main__':
    main()
