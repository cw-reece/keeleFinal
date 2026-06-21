from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.kg.cache import SliceCache, SliceCacheKey, stable_hash
from src.kg.conceptnet_store import ConceptNetStore
from src.kg.entity_extract import extract_entities
from src.kg.relation_filter import relation_set


def _tokenize(text: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True)
class SliceConfig:
    hop_depth: int = 1
    top_k: int = 10
    relation_set: str = "strict"
    min_weight: float = 0.0
    neighbor_limit: int = 200
    max_entities: int = 6
    max_ngram: int = 3
    scorer_version: str = "v1"
    random_slice: bool = False


def config_hash(cfg: SliceConfig) -> str:
    return stable_hash({
        "hop_depth": cfg.hop_depth,
        "top_k": cfg.top_k,
        "relation_set": cfg.relation_set,
        "min_weight": cfg.min_weight,
        "neighbor_limit": cfg.neighbor_limit,
        "max_entities": cfg.max_entities,
        "max_ngram": cfg.max_ngram,
        "scorer_version": cfg.scorer_version,
        "random_slice": cfg.random_slice,
    })


def score_fact(question_tokens: set[str], head: str, rel: str, tail: str, weight: float) -> float:
    fact_tokens = _tokenize(f"{head} {rel} {tail}")
    inter = len(question_tokens & fact_tokens)
    union = len(question_tokens | fact_tokens) or 1
    jacc = inter / union
    return float(weight) + 0.75 * jacc + 0.05 * inter


def _random_seeds(
    store: ConceptNetStore,
    n: int,
    rng: random.Random,
) -> List[str]:

    pool = getattr(store, "all_concepts", None)
    if pool and len(pool) >= n:
        return rng.sample(pool, n)

    if hasattr(store, "get_random_concepts"):
        return store.get_random_concepts(n, rng)

    fallback = [
        "dog", "water", "tree", "car", "book", "food", "table", "chair",
        "house", "light", "fire", "stone", "bird", "fish", "road", "clock",
        "paper", "music", "plant", "river",
    ]
    return rng.sample(fallback, min(n, len(fallback)))


def build_slice(
    *,
    store: ConceptNetStore,
    cache: SliceCache,
    question_id: int,
    image_id: int,
    question_text: str,
    cfg: SliceConfig,
) -> Tuple[Dict[str, Any], bool]:
    """Build (or load) a cached KG slice for one (image, question). Returns (slice_obj, cache_hit)."""
    ch = config_hash(cfg)
    key = SliceCacheKey(config_hash=ch, question_id=int(question_id), image_id=int(image_id))
    cached = cache.load(key)
    if cached is not None:
        return cached, True

    t0 = time.time()
    rels = relation_set(cfg.relation_set)


    if cfg.random_slice:


        rng = random.Random(int(question_id) * 31337 + int(image_id))
        seeds = _random_seeds(store, cfg.max_entities, rng)
        q_tokens = set()
    else:
        ex = extract_entities(question_text, max_entities=cfg.max_entities, max_ngram=cfg.max_ngram)
        seeds = ex.entities
        q_tokens = set(ex.tokens) | _tokenize(question_text)

    candidates: List[Dict[str, Any]] = []


    for s in seeds:
        for e in store.get_neighbors(
            s,
            relation_whitelist=rels,
            min_weight=cfg.min_weight,
            limit=cfg.neighbor_limit,
        ):
            head = s
            tail = e.other
            sc = score_fact(q_tokens, head, e.relation, tail, e.weight)
            candidates.append({
                "head": head,
                "relation": e.relation,
                "tail": tail,
                "weight": float(e.weight),
                "score": float(sc),
                "surface": e.surface,
                "hop": 1,
                "seed": s,
            })


    if cfg.hop_depth >= 2 and candidates:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        expand_nodes = [c["tail"] for c in candidates[: min(len(candidates), cfg.top_k)]]
        for node in expand_nodes:
            for e in store.get_neighbors(
                node,
                relation_whitelist=rels,
                min_weight=cfg.min_weight,
                limit=max(50, cfg.neighbor_limit // 2),
            ):
                head = node
                tail = e.other
                sc = score_fact(q_tokens, head, e.relation, tail, e.weight) * 0.9
                candidates.append({
                    "head": head,
                    "relation": e.relation,
                    "tail": tail,
                    "weight": float(e.weight),
                    "score": float(sc),
                    "surface": e.surface,
                    "hop": 2,
                    "seed": node,
                })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[: cfg.top_k]

    dt_ms = (time.time() - t0) * 1000.0
    slice_obj: Dict[str, Any] = {
        "question_id": int(question_id),
        "image_id": int(image_id),
        "question_text": question_text,
        "entities": seeds,
        "facts": top,
        "stats": {
            "cache_hit": False,
            "hop_depth": cfg.hop_depth,
            "top_k": cfg.top_k,
            "relation_set": cfg.relation_set,
            "min_weight": cfg.min_weight,
            "neighbor_limit": cfg.neighbor_limit,
            "n_entities": len(seeds),
            "n_candidates": len(candidates),
            "n_facts": len(top),
            "build_ms": dt_ms,
            "config_hash": ch,
            "random_slice": cfg.random_slice,
        },
    }

    cache.save(key, slice_obj)
    return slice_obj, False
