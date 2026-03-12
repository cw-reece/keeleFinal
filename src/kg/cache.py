# src/kg/cache.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def stable_hash(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SliceCacheKey:
    config_hash: str
    question_id: int
    image_id: int


class SliceCache:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def path_for(self, key: SliceCacheKey) -> Path:
        d = self.root_dir / key.config_hash
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key.question_id}_{key.image_id}.json"

    def load(self, key: SliceCacheKey) -> Optional[Dict[str, Any]]:
        p = self.path_for(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def save(self, key: SliceCacheKey, obj: Dict[str, Any]) -> Path:
        p = self.path_for(key)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        return p
