# scripts/build_conceptnet_db.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.kg.conceptnet_ingest import ingest_to_sqlite
from src.utils.config import load_config


def _cfg_get(cfg: dict, path: list[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit_lines", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    assertions = _cfg_get(cfg, ["conceptnet", "assertions_path"])
    db_path = _cfg_get(cfg, ["conceptnet", "db_path"])
    min_weight = float(_cfg_get(cfg, ["conceptnet", "min_weight"], 0.0))

    if not assertions or not db_path:
        raise SystemExit("Missing conceptnet.assertions_path or conceptnet.db_path in config")

    assertions_p = Path(assertions)
    db_p = Path(db_path)

    if db_p.exists() and not args.force:
        print("DB exists, skipping. Use --force to rebuild:", db_p)
        return 0

    if db_p.exists() and args.force:
        db_p.unlink()

    stats = ingest_to_sqlite(
        assertions_p,
        db_p,
        min_weight=min_weight,
        bidirectional=True,
        batch_size=int(_cfg_get(cfg, ["conceptnet", "batch_size"], 50_000)),
        limit_lines=args.limit_lines,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
