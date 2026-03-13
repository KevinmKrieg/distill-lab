from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import torch

from src.eval import load_model_from_checkpoint
from src.utils import load_config, pick_device, set_seed


@dataclass
class BenchmarkResult:
    label: str
    checkpoint: str
    device: str
    measured_batches: int
    measured_tokens: int
    average_batch_latency_ms: float
    average_sample_latency_ms: float
    average_token_latency_us: float
    tokens_per_second: float
    peak_device_memory_mb: float | None
    peak_process_rss_mb: float | None


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def current_process_rss_mb() -> float | None:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return float(output.strip()) / 1024.0
    except Exception:
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return rss / (1024 * 1024)
            return rss / 1024.0
        except Exception:
            return None


def benchmark_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    label: str | None = None,
    device: torch.device | None = None,
    warmup_batches: int = 5,
    benchmark_batches: int = 20,
) -> BenchmarkResult:
    config = load_config(str(config_path))
    set_seed(config["seed"])
    device = device or pick_device()
    model, loaders = load_model_from_checkpoint(config, str(checkpoint_path), device)
    loader = loaders["val"]

    model.eval()
    device_peak_mb: float | None = None
    process_peak_mb = current_process_rss_mb()
    measured_batches = 0
    measured_tokens = 0
    measured_samples = 0
    total_seconds = 0.0

    iterator = iter(loader)
    with torch.no_grad():
        for _ in range(warmup_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            _ = model(input_ids, attention_mask)
        synchronize_device(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(benchmark_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            token_count = int(batch.labels.ne(loaders["pad_id"]).sum().item())
            sample_count = int(batch.input_ids.size(0))

            synchronize_device(device)
            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            synchronize_device(device)
            elapsed = time.perf_counter() - start

            total_seconds += elapsed
            measured_batches += 1
            measured_tokens += token_count
            measured_samples += sample_count

            rss_mb = current_process_rss_mb()
            if rss_mb is not None:
                process_peak_mb = rss_mb if process_peak_mb is None else max(process_peak_mb, rss_mb)

            if device.type == "cuda":
                device_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
                current_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                device_peak_mb = current_mb if device_peak_mb is None else max(device_peak_mb, current_mb)

    average_batch_latency_ms = (total_seconds / max(measured_batches, 1)) * 1000.0
    average_sample_latency_ms = (total_seconds / max(measured_samples, 1)) * 1000.0
    average_token_latency_us = (total_seconds / max(measured_tokens, 1)) * 1_000_000.0
    tokens_per_second = measured_tokens / max(total_seconds, 1e-9)

    return BenchmarkResult(
        label=label or Path(checkpoint_path).parent.name,
        checkpoint=str(checkpoint_path),
        device=str(device),
        measured_batches=measured_batches,
        measured_tokens=measured_tokens,
        average_batch_latency_ms=average_batch_latency_ms,
        average_sample_latency_ms=average_sample_latency_ms,
        average_token_latency_us=average_token_latency_us,
        tokens_per_second=tokens_per_second,
        peak_device_memory_mb=device_peak_mb,
        peak_process_rss_mb=process_peak_mb,
    )


def save_benchmark(result: BenchmarkResult, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


def load_benchmark(path: str | Path) -> Dict | None:
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
