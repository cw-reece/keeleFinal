# scripts/run_matrix.py
"""Run an experiment matrix by generating per-run YAML configs and invoking train scripts.

Usage:
  python -m scripts.run_matrix --matrix configs/experiment_matrix.yaml
  python -m scripts.run_matrix --matrix configs/experiment_matrix.yaml --dry_run

Notes:
- Creates configs under configs/generated/matrix/<run_tag>.yaml
- Resume-friendly: re-running will just re-run commands; use unique tags to avoid overwriting.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def deep_set(d: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def make_tag(prefix: str, overrides: Dict[str, Any]) -> str:
    bits = []
    for k in sorted(overrides.keys()):
        v = overrides[k]
        k2 = k.split(".")[-1]
        v2 = str(v).replace("/", "-")
        bits.append(f"{k2}{v2}")
    suffix = "_".join(bits).replace(" ", "")
    return f"{prefix}_{suffix}" if suffix else prefix


def product_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = []
    for prod in itertools.product(*vals):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    matrix = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))

    base_config_path = Path(matrix["base_config"])
    base_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

    runner = matrix.get("runner", "fusion")  # fusion | baseline
    prefix = matrix.get("tag_prefix", "m6")
    gen_dir = Path(matrix.get("generated_config_dir", "configs/generated/matrix"))
    gen_dir.mkdir(parents=True, exist_ok=True)

    grid = matrix.get("grid", {})
    combos = product_grid(grid) if grid else [{}]

    fixed = matrix.get("fixed", {})
    limits = matrix.get("limits", {})

    base_cfg2 = deepcopy(base_cfg)
    for k, v in fixed.items():
        deep_set(base_cfg2, k, v)
    for k, v in limits.items():
        deep_set(base_cfg2, k, v)

    module = matrix.get("run_module") or ("src.train_fusion" if runner == "fusion" else "src.train_baseline")

    print(f"Matrix combos: {len(combos)}")
    print(f"Runner: {runner}")
    print(f"Module: {module}")
    print(f"Base config: {base_config_path}")

    for i, overrides in enumerate(combos, 1):
        cfg = deepcopy(base_cfg2)
        for k, v in overrides.items():
            deep_set(cfg, k, v)

        tag = make_tag(prefix, overrides)
        cfg_path = gen_dir / f"{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        cmd = ["python", "-m", module, "--config", str(cfg_path), "--tag", tag]
        print(f"[{i}/{len(combos)}] {tag}")
        print(" ", " ".join(cmd))

        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
