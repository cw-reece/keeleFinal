from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class NeighborEdge:
    concept: str
    relation: str
    other: str
    weight: float
    surface: str


class ConceptNetStore:
    """SQLite-backed neighbor lookup for ConceptNet (English-only index)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_random_concepts(self, n: int, rng: random.Random) -> List[str]:
        """
        Return n concept strings sampled from the DB.
        Loads a pool of 2000 concepts once and caches it on the instance,
        then samples from that pool using the caller's rng (deterministic per-call).
        """
        if not hasattr(self, "_concept_pool"):
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT DISTINCT concept FROM edges ORDER BY RANDOM() LIMIT 2000"
                ).fetchall()
                self._concept_pool = [r[0] for r in rows]
            finally:
                conn.close()
        return rng.sample(self._concept_pool, min(n, len(self._concept_pool)))

    def get_neighbors(
        self,
        concept: str,
        *,
        relation_whitelist: Optional[set[str]] = None,
        min_weight: float = 0.0,
        limit: int = 200,
    ) -> List[NeighborEdge]:
        q = "SELECT concept, relation, other, weight, surface FROM edges WHERE concept=? AND weight>=?"
        params: list = [concept, float(min_weight)]

        if relation_whitelist:
            ph = ",".join(["?"] * len(relation_whitelist))
            q += f" AND relation IN ({ph})"
            params.extend(sorted(list(relation_whitelist)))

        q += " ORDER BY weight DESC LIMIT ?"
        params.append(int(limit))

        conn = self._connect()
        try:
            cur = conn.execute(q, params)
            rows = cur.fetchall()
            return [
                NeighborEdge(
                    concept=row["concept"],
                    relation=row["relation"],
                    other=row["other"],
                    weight=float(row["weight"]),
                    surface=str(row["surface"] or ""),
                )
                for row in rows
            ]
        finally:
            conn.close()
