from __future__ import annotations

import argparse
import time

from run_metadata import init_run, write_metrics
from src.datasets.okvqa import OKVQADataset
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
    ap.add_argument("--tag", default="train")
    ap.add_argument("--dry_run", type=str, default="true")
    ap.add_argument("--limit", type=int, default=32)
    args = ap.parse_args()

    dry_run = args.dry_run.lower() in ("1", "true", "yes", "y")

    cfg = load_config(args.config)
    out_root = _cfg_get(cfg, ["logging", "out_root"], "experiments/runs")

    ctx = init_run(
        tag=args.tag,
        config_path=args.config,
        out_root=out_root,
        include_pip_freeze=bool(_cfg_get(cfg, ["logging", "include_pip_freeze"], False)),
        extra={"dry_run": dry_run, "limit": args.limit},
    )

    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])
    train_q = _cfg_get(cfg, ["data", "train_questions_json"])
    train_a = _cfg_get(cfg, ["data", "train_annotations_json"])

    if not all([ann_dir, img_root, train_q, train_a]):
        write_metrics(ctx.run_dir, {"status": "error", "error": "Missing data.* fields for train split."})
        return 2

    ds = OKVQADataset(
        questions_path=f"{ann_dir}/{train_q}",
        annotations_path=f"{ann_dir}/{train_a}",
        images_root=img_root,
        load_images=False,
    )

    n = min(len(ds), args.limit)
    t0 = time.time()
    for i in range(n):
        _ = ds[i]
    dt = time.time() - t0

    write_metrics(
        ctx.run_dir,
        {
            "status": "ok",
            "dry_run": dry_run,
            "n_examples_iterated": n,
            "seconds": dt,
            "message": "Training stub executed. Next: implement real forward/backward + loss.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
