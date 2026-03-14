# scripts/summarize_runs.py
"""Summarize metrics.json across runs into CSV + Markdown.

Usage:
  python -m scripts.summarize_runs --out_dir reports --filter m5_ --filter m6_
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


FIELDS = [
    "run_id",
    "fusion_mode",
    "topn_rerank",
    "temperature",
    "alpha",
    "baseline_acc",
    "fused_acc",
    "delta_acc",
    "kg_relation_set",
    "kg_top_k",
    "kg_hop_depth",
    "seconds",
]


def jget(d: dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="experiments/runs")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--filter", action="append", default=[], help="Include run_ids containing this substring (repeatable)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue
        rid = run_path.name
        if args.filter and not any(f in rid for f in args.filter):
            continue

        mp = run_path / "metrics.json"
        if not mp.exists():
            continue

        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue

        base_acc = m.get("val_vqa_soft_accuracy_mean")
        fused_acc = None
        delta = None
        alpha = None

        if isinstance(jget(m, "val_baseline", "val_vqa_soft_accuracy_mean"), (int, float)):
            base_acc = jget(m, "val_baseline", "val_vqa_soft_accuracy_mean")
            fused_acc = jget(m, "val_fused", "val_vqa_soft_accuracy_mean")
            delta = m.get("delta_val_acc")
            alpha = jget(m, "alpha_info", "alpha")

        row = {
            "run_id": rid,
            "fusion_mode": m.get("fusion_mode"),
            "topn_rerank": m.get("topn_rerank"),
            "temperature": m.get("temperature"),
            "alpha": alpha,
            "baseline_acc": base_acc,
            "fused_acc": fused_acc,
            "delta_acc": delta,
            "kg_relation_set": jget(m, "slice_config", "relation_set") or jget(m, "config", "slice_config", "relation_set"),
            "kg_top_k": jget(m, "slice_config", "top_k") or jget(m, "config", "slice_config", "top_k"),
            "kg_hop_depth": jget(m, "slice_config", "hop_depth") or jget(m, "config", "slice_config", "hop_depth"),
            "seconds": m.get("seconds"),
        }
        rows.append(row)

    csv_path = out_dir / "run_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})

    md_path = out_dir / "run_summary.md"
    lines = []
    lines.append("# Run Summary\n\n")
    lines.append("| " + " | ".join(FIELDS) + " |\n")
    lines.append("| " + " | ".join(["---"] * len(FIELDS)) + " |\n")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in FIELDS) + " |\n")
    md_path.write_text("".join(lines), encoding="utf-8")

    print("Wrote:", csv_path)
    print("Wrote:", md_path)
    print("Rows:", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
