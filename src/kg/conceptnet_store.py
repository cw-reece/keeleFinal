# src/kg/conceptnet_store.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


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
            # generate (?, ?, ...) placeholders
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
