# src/kg/entity_extract.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

_STOPWORDS = {
    "a","an","the","of","in","on","at","to","for","from","by","with","about","as","and","or","but",
    "is","are","was","were","be","been","being","do","does","did","doing","can","could","would","should",
    "what","which","who","whom","whose","when","where","why","how",
    "this","that","these","those","it","its","they","them","their","there",
    "i","you","we","he","she","him","her","my","your","our",
    "not","no","yes","than","then","too","very","into","up","down","over","under",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_concept(text: str) -> str:
    """Normalize to ConceptNet-like token form: lowercase, spaces->underscore."""
    t = text.strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_]", "", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


@dataclass(frozen=True)
class EntityExtraction:
    entities: List[str]
    tokens: List[str]


def extract_entities(question: str, *, max_entities: int = 6, max_ngram: int = 3) -> EntityExtraction:
    """Cheap, deterministic entity extraction (no external NLP deps).

    Strategy:
    - tokenize alnum
    - drop stopwords
    - generate ngrams up to max_ngram
    - prefer longer ngrams (3 > 2 > 1), then earlier appearance
    """
    raw_tokens = _TOKEN_RE.findall(question.lower())
    tokens = [t for t in raw_tokens if t and t not in _STOPWORDS]

    # Build candidate ngrams in order of appearance.
    candidates: list[str] = []
    n = len(tokens)
    for k in range(max_ngram, 0, -1):
        for i in range(0, n - k + 1):
            phrase = "_".join(tokens[i:i+k])
            phrase = normalize_concept(phrase)
            if phrase and phrase not in _STOPWORDS:
                candidates.append(phrase)

    # Deduplicate while preserving order.
    seen: Set[str] = set()
    entities: List[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        entities.append(c)
        if len(entities) >= max_entities:
            break

    return EntityExtraction(entities=entities, tokens=tokens)
