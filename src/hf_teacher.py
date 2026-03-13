from __future__ import annotations

from typing import Dict

import torch
from torch import nn

try:
    from transformers import AutoModelForCausalLM
except Exception:  # pragma: no cover - handled at runtime
    AutoModelForCausalLM = None


class HFCausalLMTeacher(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        config = model.config
        self.hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
        self.vocab_size = int(config.vocab_size)
        if self.hidden_size is None:
            raise ValueError("Could not resolve hidden size from Hugging Face model config.")

    @classmethod
    def from_pretrained(cls, config: Dict, device: torch.device) -> "HFCausalLMTeacher":
        if AutoModelForCausalLM is None:
            raise ImportError("transformers is required for Hugging Face teacher distillation. Install it with `pip install transformers`.")
        model_name = config["model_name_or_path"]
        kwargs = {
            "cache_dir": config.get("cache_dir"),
            "revision": config.get("revision"),
            "trust_remote_code": config.get("trust_remote_code", False),
            "local_files_only": config.get("local_files_only", False),
        }
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return cls(model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        return {"logits": outputs.logits, "hidden_states": hidden_states}
