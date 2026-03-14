# src/kg/text_encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


@dataclass
class EncodeResult:
    embeddings: torch.Tensor  # [N,H] on CPU
    dim: int


class TextEncoder:
    """HF encoder + mean pooling + L2 norm (fast, dependency-light sentence embeddings)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 64, desc: str = "encode") -> EncodeResult:
        all_vecs: List[torch.Tensor] = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
            chunk = list(texts[i : i + batch_size])
            enc = self.tok(chunk, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
            all_vecs.append(vec.detach().cpu())
        embs = torch.cat(all_vecs, dim=0) if all_vecs else torch.empty((0, 0))
        dim = int(embs.shape[-1]) if embs.numel() else 0
        return EncodeResult(embeddings=embs, dim=dim)
