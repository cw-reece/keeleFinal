# src/kg/conceptnet_ingest.py
from __future__ import annotations

import gzip
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple
from urllib.parse import unquote

from tqdm import tqdm


@dataclass(frozen=True)
class EdgeRow:
    concept: str
    relation: str
    other: str
    weight: float
    surface: str


def _extract_en_concept(uri: str) -> Optional[str]:
    """Return canonical English concept string from ConceptNet URI, else None.

    Examples:
      /c/en/dog -> dog
      /c/en/hot_dog/n -> hot_dog
    """
    if not uri.startswith("/c/en/"):
        return None
    rest = uri[len("/c/en/"):]
    # stop at next '/'
    term = rest.split("/", 1)[0]
    term = unquote(term)
    term = term.strip().lower()
    if not term:
        return None
    return term


def _relation_name(uri: str) -> str:
    # /r/IsA -> IsA
    if uri.startswith("/r/"):
        return uri[len("/r/"):]
    return uri


def _parse_weight(meta_json: str) -> float:
    try:
        meta = json.loads(meta_json)
        w = meta.get("weight", 1.0)
        return float(w)
    except Exception:
        return 1.0


def iter_assertions(path: Path) -> Iterator[Tuple[str, str, str, float, str]]:
    """Yield (head, relation, tail, weight, surface) for English-only assertions."""
    opener = gzip.open if path.suffix.endswith("gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # ConceptNet assertions are tab-separated. Format:
            # uri, rel, start, end, meta_json
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            rel_uri = parts[1]
            start_uri = parts[2]
            end_uri = parts[3]
            meta_json = parts[4]

            head = _extract_en_concept(start_uri)
            tail = _extract_en_concept(end_uri)
            if head is None or tail is None:
                continue

            rel = _relation_name(rel_uri)
            w = _parse_weight(meta_json)

            # surface text is nice for audit, but optional
            surface = ""
            try:
                meta = json.loads(meta_json)
                surface = str(meta.get("surfaceText") or "")
            except Exception:
                surface = ""

            yield head, rel, tail, w, surface


def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS edges (
            concept TEXT NOT NULL,
            relation TEXT NOT NULL,
            other   TEXT NOT NULL,
            weight  REAL NOT NULL,
            surface TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_edges_concept ON edges(concept);
        CREATE INDEX IF NOT EXISTS idx_edges_concept_rel ON edges(concept, relation);
        """
    )
    conn.commit()


def ingest_to_sqlite(
    assertions_path: Path,
    db_path: Path,
    *,
    min_weight: float = 0.0,
    bidirectional: bool = True,
    batch_size: int = 50_000,
    limit_lines: Optional[int] = None,
) -> dict:
    """Build/append an English-only ConceptNet edge index in SQLite.

    Stores rows as (concept, relation, other, weight, surface).
    If bidirectional=True, inserts both head->tail and tail->head.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    create_schema(conn)

    cur = conn.cursor()
    insert_sql = "INSERT INTO edges (concept, relation, other, weight, surface) VALUES (?, ?, ?, ?, ?)"

    t0 = time.time()
    n_in = 0
    n_out = 0
    batch = []

    it = iter_assertions(assertions_path)
    if limit_lines is not None:
        it = (x for _, x in zip(range(limit_lines), it))  # type: ignore

    for head, rel, tail, w, surface in tqdm(it, desc="ingest ConceptNet (en)", unit="edges"):
        n_in += 1
        if w < min_weight:
            continue

        batch.append((head, rel, tail, w, surface))
        if bidirectional:
            batch.append((tail, rel, head, w, surface))

        if len(batch) >= batch_size:
            cur.executemany(insert_sql, batch)
            conn.commit()
            n_out += len(batch)
            batch.clear()

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()
        n_out += len(batch)
        batch.clear()

    dt = time.time() - t0
    conn.close()

    return {
        "status": "ok",
        "assertions_path": str(assertions_path),
        "db_path": str(db_path),
        "min_weight": min_weight,
        "bidirectional": bidirectional,
        "rows_inserted": n_out,
        "assertions_used": n_in,
        "seconds": dt,
    }
