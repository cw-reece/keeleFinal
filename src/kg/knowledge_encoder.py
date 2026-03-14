# src/kg/knowledge_encoder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.kg.text_encoder import TextEncoder


@dataclass
class KGEncoding:
    kg_emb: torch.Tensor     # [B,H]
    kg_logits: torch.Tensor  # [B,V]
    debug: Dict[str, Any]


def _facts_to_texts(facts: List[Dict[str, Any]]) -> List[str]:
    return [f"{f.get('head','')} {f.get('relation','')} {f.get('tail','')}".strip() for f in facts]


def _weighted_avg(vectors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.clamp_min(0.0)
    if float(w.sum()) <= 0:
        return vectors.mean(dim=0)
    w = w / (w.sum() + 1e-8)
    return (vectors * w.unsqueeze(-1)).sum(dim=0)


class AnswerEmbeddingCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, model_name: str, vocab_size: int) -> Path:
        safe = model_name.replace("/", "__")
        return self.cache_dir / f"answer_emb__{safe}__{vocab_size}.pt"

    def load(self, model_name: str, answers: List[str]) -> Optional[torch.Tensor]:
        p = self.cache_path(model_name, len(answers))
        if not p.exists():
            return None
        obj = torch.load(p, map_location="cpu")
        if obj.get("model_name") != model_name:
            return None
        if obj.get("answers") != answers:
            return None
        emb = obj.get("embeddings")
        return emb if isinstance(emb, torch.Tensor) else None

    def save(self, model_name: str, answers: List[str], embeddings: torch.Tensor) -> Path:
        p = self.cache_path(model_name, len(answers))
        torch.save({"model_name": model_name, "answers": answers, "embeddings": embeddings}, p)
        return p


class KnowledgeEncoder:
    """Facts -> pooled knowledge embedding -> logits over answer vocab.

    kg_logits[a] is cosine_sim(kg_emb, answer_emb[a]) * temperature.
    """

    def __init__(
        self,
        *,
        embedding_model: str,
        answers: List[str],
        device: str,
        cache_dir: str | Path,
        temperature: float = 10.0,
        fact_batch_size: int = 64,
        answer_batch_size: int = 256,
    ) -> None:
        self.embedding_model = embedding_model
        self.device = device
        self.temperature = float(temperature)
        self.fact_batch_size = int(fact_batch_size)
        self.answer_batch_size = int(answer_batch_size)

        self.encoder = TextEncoder(embedding_model, device=device)
        self.answers = answers

        self.ans_cache = AnswerEmbeddingCache(cache_dir)
        cached = self.ans_cache.load(embedding_model, answers)
        if cached is None:
            res = self.encoder.encode(answers, batch_size=self.answer_batch_size, desc="encode answers")
            self.answer_emb = res.embeddings
            self.ans_cache.save(embedding_model, answers, self.answer_emb)
        else:
            self.answer_emb = cached

        self.answer_emb = torch.nn.functional.normalize(self.answer_emb, p=2, dim=-1)

    @torch.no_grad()
    def encode_batch(self, slices: List[Dict[str, Any]]) -> KGEncoding:
        V, H = int(self.answer_emb.shape[0]), int(self.answer_emb.shape[1])
        ans = self.answer_emb.to(self.device)

        kg_embs = []
        kg_logits = []
        debug: Dict[str, Any] = {"empty_count": 0}

        for s in slices:
            facts = list(s.get("facts", []))
            if not facts:
                debug["empty_count"] += 1
                emb = torch.zeros((H,), device=self.device)
                emb = torch.nn.functional.normalize(emb + 1e-8, p=2, dim=-1)
                logits = torch.zeros((V,), device=self.device)
                kg_embs.append(emb)
                kg_logits.append(logits)
                continue

            fact_texts = _facts_to_texts(facts)
            w = torch.tensor([float(f.get("score", 0.0)) for f in facts], device=self.device)
            w = torch.nn.functional.relu(w)

            res = self.encoder.encode(fact_texts, batch_size=self.fact_batch_size, desc="encode facts")
            fact_emb = torch.nn.functional.normalize(res.embeddings.to(self.device), p=2, dim=-1)

            emb = torch.nn.functional.normalize(_weighted_avg(fact_emb, w), p=2, dim=-1)

            sims = emb @ ans.T  # [V]
            logits = sims * self.temperature

            kg_embs.append(emb)
            kg_logits.append(logits)

        return KGEncoding(
            kg_emb=torch.stack(kg_embs, dim=0),
            kg_logits=torch.stack(kg_logits, dim=0),
            debug=debug,
        )
