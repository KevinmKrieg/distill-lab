from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def masked_language_model_loss(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=pad_id)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int,
    temperature: float,
    alpha: float,
    student_hidden: Optional[torch.Tensor] = None,
    teacher_hidden: Optional[torch.Tensor] = None,
    hidden_loss_weight: float = 0.0,
    hidden_projector: Optional[torch.nn.Module] = None,
) -> Dict[str, torch.Tensor]:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    valid_mask = labels.ne(pad_id)
    kl = (token_kl * valid_mask).sum() / valid_mask.sum().clamp_min(1)
    kl = kl * (temperature ** 2)
    ce = masked_language_model_loss(student_logits, labels, pad_id)
    total = alpha * kl + (1.0 - alpha) * ce

    hidden_mse = torch.zeros((), device=student_logits.device)
    if (
        hidden_loss_weight > 0.0
        and student_hidden is not None
        and teacher_hidden is not None
        and hidden_projector is not None
    ):
        projected = hidden_projector(student_hidden)
        mask = labels.ne(pad_id).unsqueeze(-1)
        diff = (projected - teacher_hidden) * mask
        denom = mask.sum().clamp_min(1)
        hidden_mse = diff.pow(2).sum() / denom
        total = total + hidden_loss_weight * hidden_mse

    return {"loss": total, "kl": kl.detach(), "ce": ce.detach(), "hidden_mse": hidden_mse.detach()}
