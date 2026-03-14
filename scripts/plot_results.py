# scripts/plot_results.py
"""Make basic plots from reports/run_summary.csv.

Usage:
  python -m scripts.plot_results --csv reports/run_summary.csv --out_dir reports/plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def ffloat(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/run_summary.csv")
    ap.add_argument("--out_dir", default="reports/plots")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    fx = [x for x in rows if x.get("delta_acc") not in (None, "", "None")]

    deltas = [ffloat(x.get("delta_acc")) for x in fx]
    deltas = [d for d in deltas if d is not None]

    if deltas:
        plt.figure()
        plt.hist(deltas, bins=20)
        plt.title("Delta accuracy distribution (fused - baseline)")
        plt.xlabel("delta_acc")
        plt.ylabel("count")
        p = out_dir / "delta_hist.png"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print("Wrote:", p)

    topk = []
    d2 = []
    for x in fx:
        tk = ffloat(x.get("kg_top_k"))
        da = ffloat(x.get("delta_acc"))
        if tk is None or da is None:
            continue
        topk.append(tk)
        d2.append(da)

    if d2:
        plt.figure()
        plt.scatter(topk, d2)
        plt.title("Delta accuracy vs top_k")
        plt.xlabel("top_k")
        plt.ylabel("delta_acc")
        p = out_dir / "delta_vs_topk.png"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print("Wrote:", p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
