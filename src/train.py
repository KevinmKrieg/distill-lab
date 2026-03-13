from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from src.data import create_dataloaders
from src.eval import evaluate_model
from src.hf_teacher import HFCausalLMTeacher
from src.losses import distillation_loss, masked_language_model_loss
from src.models import build_model
from src.utils import append_jsonl, ensure_dir, load_config, pick_device, save_config, set_seed


def build_teacher_for_distillation(config: Dict, vocab_size: int, max_seq_len: int, device: torch.device) -> nn.Module:
    distill_cfg = config["distillation"]
    teacher_source = distill_cfg.get("teacher_source", "local")

    if teacher_source == "huggingface":
        teacher_cfg = distill_cfg["teacher_hf"]
        teacher = HFCausalLMTeacher.from_pretrained(teacher_cfg, device)
        if teacher.vocab_size != vocab_size:
            raise ValueError(
                "Teacher vocab size does not match student/data vocab size. "
                "When using a Hugging Face teacher, set `data.tokenizer.level: hf` with the same tokenizer family."
            )
        return teacher

    teacher = build_model(vocab_size, max_seq_len, config["teacher_model"]).to(device)
    checkpoint = torch.load(distill_cfg["teacher_checkpoint"], map_location=device)
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def build_training_stack(config: Dict, loaders: Dict, device: torch.device) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    mode = config["training"]["mode"]
    max_seq_len = config["data"]["max_seq_len"] - 1
    vocab_size = loaders["vocab_size"]

    if mode == "distill":
        student = build_model(vocab_size, max_seq_len, config["student_model"]).to(device)
        teacher = build_teacher_for_distillation(config, vocab_size, max_seq_len, device)
        hidden_projector = None
        if config["distillation"]["hidden_loss_weight"] > 0.0:
            hidden_projector = nn.Linear(
                config["student_model"]["d_model"],
                teacher.hidden_size,
            ).to(device)
        return student, teacher, hidden_projector

    model = build_model(vocab_size, max_seq_len, config["model"]).to(device)
    return model, None, None


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: Dict,
    hidden_projector: Optional[nn.Module] = None,
    is_best: bool = False,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_config,
    }
    if hidden_projector is not None:
        payload["hidden_projector_state"] = hidden_projector.state_dict()

    last_path = output_dir / "last.pt"
    torch.save(payload, last_path)
    if is_best:
        torch.save(payload, output_dir / "best.pt")


def train_one_epoch(
    model: nn.Module,
    teacher: Optional[nn.Module],
    hidden_projector: Optional[nn.Module],
    loader,
    optimizer: torch.optim.Optimizer,
    config: Dict,
    device: torch.device,
    pad_id: int,
) -> Dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    running = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "hidden_mse": 0.0}

    for step, batch in enumerate(loader, start=1):
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        attention_mask = batch.attention_mask.to(device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids, attention_mask)
        if teacher is None:
            loss = masked_language_model_loss(outputs["logits"], labels, pad_id)
            metrics = {
                "loss": loss.detach(),
                "kl": torch.zeros((), device=device),
                "ce": loss.detach(),
                "hidden_mse": torch.zeros((), device=device),
            }
        else:
            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask)
            metrics = distillation_loss(
                student_logits=outputs["logits"],
                teacher_logits=teacher_outputs["logits"],
                labels=labels,
                pad_id=pad_id,
                temperature=config["distillation"]["temperature"],
                alpha=config["distillation"]["alpha"],
                student_hidden=outputs["hidden_states"],
                teacher_hidden=teacher_outputs["hidden_states"],
                hidden_loss_weight=config["distillation"]["hidden_loss_weight"],
                hidden_projector=hidden_projector,
            )
            loss = metrics["loss"]

        loss.backward()
        if hidden_projector is not None:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(hidden_projector.parameters()), config["training"]["grad_clip"])
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
        optimizer.step()

        for key in running:
            running[key] += metrics[key].item()

        if step % config["training"]["log_every"] == 0:
            avg_loss = running["loss"] / step
            print(f"step={step} loss={avg_loss:.4f}")

    total_steps = max(len(loader), 1)
    return {key: value / total_steps for key, value in running.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small teacher/student language model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = pick_device()
    loaders = create_dataloaders(config)
    output_dir = ensure_dir(config["output_dir"])
    metrics_path = output_dir / "metrics.jsonl"
    save_config(output_dir / "run_config.yaml", config)

    model, teacher, hidden_projector = build_training_stack(config, loaders, device)
    params = list(model.parameters())
    if hidden_projector is not None:
        params += list(hidden_projector.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_perplexity = math.inf
    model_config = config["student_model"] if config["training"]["mode"] == "distill" else config["model"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            teacher=teacher,
            hidden_projector=hidden_projector,
            loader=loaders["train"],
            optimizer=optimizer,
            config=config,
            device=device,
            pad_id=loaders["pad_id"],
        )
        perplexity, accuracy = evaluate_model(model, loaders["val"], device, loaders["pad_id"])
        is_best = perplexity < best_perplexity
        if is_best:
            best_perplexity = perplexity
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            model_config=model_config,
            hidden_projector=hidden_projector,
            is_best=is_best,
        )

        payload = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_kl": round(train_metrics["kl"], 4),
            "train_ce": round(train_metrics["ce"], 4),
            "train_hidden_mse": round(train_metrics["hidden_mse"], 4),
            "val_perplexity": round(perplexity, 4),
            "val_token_accuracy": round(accuracy, 4),
        }
        append_jsonl(metrics_path, payload)
        print(
            f"epoch={epoch} train_loss={payload['train_loss']:.4f} "
            f"val_perplexity={payload['val_perplexity']:.4f} "
            f"val_token_accuracy={payload['val_token_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
