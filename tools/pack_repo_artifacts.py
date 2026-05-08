# tools/pack_repo_artifacts.py
"""Collect key dissertation artifacts from the repo into a single folder + zip.

Run from repo root:
  python tools/pack_repo_artifacts.py --repo_root . --out_dir dissertation_pack

This does NOT modify your experiments; it only copies files.

By default it copies:
- docs/*.md (if present)
- configs used (baseline/kg/fusion/matrix)
- reports tables + plots
- selected error-analysis cases (selected_cases.md + summary.json)
- metrics.json for key runs (baseline freeze + fusion runs + m6 matrix runs)

It also writes MANIFEST.md with missing-file notes.

"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


DEFAULT_RUN_IDS = [
    "BASELINE_FREEZE_20260312_1456",
    "20260313_140303_m5_fusion_weighted_fullval",
    "20260313_141000_m5_gated_fullval",
    "20260313_143730_m5_topn_weighted_fullval",
    "20260313_145031_m5_topn20_weighted_fullval",
    "20260313_145705_m5_topn20_gated_fullval",
]


DEFAULT_CONFIGS = [
    "configs/baseline_train_v4_suggested.yaml",
    "configs/kg_slice.yaml",
    "configs/fusion_train_v3_topn.yaml",
    "configs/experiment_matrix.yaml",
]


DEFAULT_DOCS = [
    "docs/metrics.md",
    "docs/architecture.md",
    "docs/risks.md",
    "docs/baseline_results.md",
    "docs/baseline_results_UPDATED.md",
    "docs/fusion_results.md",
    "docs/experiment_matrix.md",
]


DEFAULT_REPORTS = [
    "reports/run_summary.csv",
    "reports/run_summary.md",
]


def copy_if_exists(repo_root: Path, rel: str, out_root: Path, copied: List[str], missing: List[str]) -> None:
    src = repo_root / rel
    if src.exists():
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    else:
        missing.append(rel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--out_dir", default="dissertation_pack")
    ap.add_argument("--include_jsonl", action="store_true", help="Also copy large predictions.jsonl files (can be huge).")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out = Path(args.out_dir).resolve()

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    missing: List[str] = []

    # docs
    for rel in DEFAULT_DOCS:
        copy_if_exists(repo, rel, out, copied, missing)

    # configs
    for rel in DEFAULT_CONFIGS:
        copy_if_exists(repo, rel, out, copied, missing)

    # reports
    for rel in DEFAULT_REPORTS:
        copy_if_exists(repo, rel, out, copied, missing)

    # plots
    plots = list((repo / "reports/plots").glob("*.png"))
    if plots:
        for p in plots:
            rel = str(p.relative_to(repo))
            copy_if_exists(repo, rel, out, copied, missing)
    else:
        missing.append("reports/plots/*.png")

    # error analysis (selected cases + summary)
    ea_dirs = sorted((repo / "reports/error_analysis").glob("*")) if (repo / "reports/error_analysis").exists() else []
    if not ea_dirs:
        missing.append("reports/error_analysis/* (no error analysis dirs found)")
    else:
        for d in ea_dirs:
            for relname in ["selected_cases.md", "summary.json"]:
                rel = str((d / relname).relative_to(repo))
                copy_if_exists(repo, rel, out, copied, missing)
            if args.include_jsonl:
                rel = str((d / "predictions.jsonl").relative_to(repo))
                copy_if_exists(repo, rel, out, copied, missing)

    # run metrics for key runs
    for rid in DEFAULT_RUN_IDS:
        rel = f"experiments/runs/{rid}/metrics.json"
        copy_if_exists(repo, rel, out, copied, missing)

    # all m6 matrix run metrics (small)
    m6_metrics = sorted(glob.glob(str(repo / "experiments/runs/*m6_matrix*/metrics.json")))
    if m6_metrics:
        for p in m6_metrics:
            rel = str(Path(p).relative_to(repo))
            copy_if_exists(repo, rel, out, copied, missing)
    else:
        missing.append("experiments/runs/*m6_matrix*/metrics.json")

    # write manifest
    manifest = out / "MANIFEST.md"
    lines = []
    lines.append("# Dissertation Pack Manifest\n\n")
    lines.append(f"Repo root: `{repo}`\n\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append("## Copied files\n\n")
    for c in copied:
        lines.append(f"- {c}\n")
    lines.append("\n## Missing / not found\n\n")
    for m in missing:
        lines.append(f"- {m}\n")
    manifest.write_text("".join(lines), encoding="utf-8")

    # zip
    zip_name = f"dissertation_pack_{datetime.now().strftime('%Y%m%d')}.zip"
    zip_path = out.parent / zip_name

    import zipfile
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_dir():
                continue
            z.write(p, arcname=str(p.relative_to(out.parent)))

    print("Created pack folder:", out)
    print("Created zip:", zip_path)
    print("Manifest:", manifest)
    print("Copied:", len(copied), "Missing:", len(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
