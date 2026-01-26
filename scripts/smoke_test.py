"""scripts/smoke_test.py

Creates a run folder and writes minimal metadata + dummy metrics.

This is a Milestone 1 artifact: proves your logging and run-folder discipline works
before you write training code.

Usage:
  python -m scripts.smoke_test --config configs/baseline.yaml --tag m1_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_metadata import init_run, write_metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config file (snapshotted into run folder).")
    ap.add_argument("--tag", required=True, help="Run tag used in run_id naming.")
    ap.add_argument("--pip_freeze", action="store_true", help="Capture pip freeze (slower, but reproducible).")
    ap.add_argument("--notes", default="Milestone 1 smoke test.", help="Short human note written to notes.md")
    args = ap.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        print(f"ERROR: config not found: {cfg}", file=sys.stderr)
        return 2

    ctx = init_run(
        tag=args.tag,
        config_path=str(cfg),
        include_pip_freeze=args.pip_freeze,
        argv=sys.argv,
        extra={
            "purpose": "m1_smoke_test",
            "notes": args.notes,
        },
    )

    metrics = {
        "status": "ok",
        "smoke_test": True,
        "message": "Run metadata + folder structure created successfully.",
    }
    write_metrics(ctx.run_dir, metrics)

    notes_path = Path(ctx.run_dir) / "notes.md"
    notes_path.write_text(
        f"""# Run Notes

- Purpose: Milestone 1 smoke test
- Notes: {args.notes}

Next: implement dataset loader + baseline pipeline (Milestone 2).
""",
        encoding="utf-8",
    )

    print(f"Created run: {ctx.run_id}")
    print(f"Run dir: {ctx.run_dir}")
    print(f"Metadata: {ctx.run_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
