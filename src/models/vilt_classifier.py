# src/models/vilt_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from transformers import ViltModel


@dataclass(frozen=True)
class ViltClassifierOutput:
    logits: torch.Tensor  # [B, num_labels]


class ViltForAnswerVocab(nn.Module):
    """ViLT backbone + linear classifier for a fixed answer vocabulary."""

    def __init__(self, backbone_checkpoint: str, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.vilt = ViltModel.from_pretrained(backbone_checkpoint)
        hidden = self.vilt.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> ViltClassifierOutput:
        out = self.vilt(**inputs)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        return ViltClassifierOutput(logits=logits)
