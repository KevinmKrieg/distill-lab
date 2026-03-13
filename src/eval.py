from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.data import create_dataloaders
from src.models import build_model
from src.utils import load_config, pick_device, set_seed


def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    pad_id: int,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    evaluated_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch.input_ids.to(device)
            labels = batch.labels.to(device)
            attention_mask = batch.attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
            total_loss += token_loss.item()
            total_tokens += labels.ne(pad_id).sum().item()

            predictions = logits.argmax(dim=-1)
            mask = labels.ne(pad_id)
            correct_tokens += predictions.eq(labels).masked_select(mask).sum().item()
            evaluated_tokens += mask.sum().item()

    average_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(average_loss)
    accuracy = correct_tokens / max(evaluated_tokens, 1)
    return perplexity, accuracy


def resolve_model_config(checkpoint: Dict, fallback_config: Dict, model_key: str) -> Dict:
    if "model_config" in checkpoint:
        return checkpoint["model_config"]
    return fallback_config[model_key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved language model checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to <output_dir>/best.pt.")
    parser.add_argument("--model-key", default="model", help="Config key to use when checkpoint lacks embedded config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = pick_device()
    loaders = create_dataloaders(config)

    checkpoint_path = args.checkpoint or str(Path(config["output_dir"]) / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = resolve_model_config(checkpoint, config, args.model_key)
    model = build_model(
        vocab_size=loaders["vocab_size"],
        max_seq_len=config["data"]["max_seq_len"] - 1,
        config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    perplexity, accuracy = evaluate_model(model, loaders["val"], device, loaders["pad_id"])
    metrics = {
        "checkpoint": checkpoint_path,
        "perplexity": round(perplexity, 4),
        "token_accuracy": round(accuracy, 4),
        "device": str(device),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
