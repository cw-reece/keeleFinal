# src/fusion/late_fusion.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionOutput:
    fused_logits: torch.Tensor
    alpha: torch.Tensor  # scalar (weighted) or [B,1] (gated)


class WeightedAddFusion(nn.Module):
    """fused = base + alpha * kg

    Fixes:
    - constrain alpha >= 0 (alpha = softplus(alpha_raw)) to prevent "inverting" KG.
    - allow learn_alpha=False for fixed alpha (no trainable params).
    """

    def __init__(self, alpha_init: float = 0.5, learn_alpha: bool = True) -> None:
        super().__init__()
        self.learn_alpha = learn_alpha
        if learn_alpha:
            a = float(max(alpha_init, 1e-6))
            raw = torch.log(torch.expm1(torch.tensor(a)))
            self.alpha_raw = nn.Parameter(raw)
        else:
            self.register_buffer("alpha_fixed", torch.tensor(float(alpha_init)), persistent=False)

    def alpha(self) -> torch.Tensor:
        if self.learn_alpha:
            return F.softplus(self.alpha_raw)
        return self.alpha_fixed

    def forward(self, base_logits: torch.Tensor, kg_logits: torch.Tensor) -> FusionOutput:
        a = self.alpha()
        fused = base_logits + a * kg_logits
        return FusionOutput(fused_logits=fused, alpha=a)


class GatedFusion(nn.Module):
    """fused = base + gate(kg_emb) * kg"""

    def __init__(self, emb_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, base_logits: torch.Tensor, kg_logits: torch.Tensor, kg_emb: torch.Tensor) -> FusionOutput:
        gate = self.mlp(kg_emb)  # [B,1]
        fused = base_logits + gate * kg_logits
        return FusionOutput(fused_logits=fused, alpha=gate)
