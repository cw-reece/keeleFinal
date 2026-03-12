# scripts/build_slices.py
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from run_metadata import init_run, write_metrics
from src.datasets.okvqa import OKVQADataset
from src.kg.cache import SliceCache
from src.kg.conceptnet_store import ConceptNetStore
from src.kg.slice_builder import SliceConfig, build_slice
from src.utils.config import load_config


def _cfg_get(cfg: dict, path: list[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize(nums: List[float]) -> Dict[str, float]:
    if not nums:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    nums_sorted = sorted(nums)
    p95 = nums_sorted[int(0.95 * (len(nums_sorted) - 1))]
    return {
        "mean": float(statistics.mean(nums_sorted)),
        "median": float(statistics.median(nums_sorted)),
        "p95": float(p95),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--tag", default="kg_slices")
    ap.add_argument("--sample_count", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # init run logging
    out_root = _cfg_get(cfg, ["logging", "out_root"], "experiments/runs")
    ctx = init_run(tag=args.tag, config_path=args.config, out_root=out_root, include_pip_freeze=False, extra={"purpose": "kg_slicing"})

    # dataset
    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])
    q_file = _cfg_get(cfg, ["data", "val_questions_json"] if args.split == "val" else ["data", "train_questions_json"])
    a_file = _cfg_get(cfg, ["data", "val_annotations_json"] if args.split == "val" else ["data", "train_annotations_json"])

    ds = OKVQADataset(f"{ann_dir}/{q_file}", f"{ann_dir}/{a_file}", img_root, load_images=False)

    # store + cache
    db_path = _cfg_get(cfg, ["conceptnet", "db_path"])
    store = ConceptNetStore(db_path)

    cache_root = _cfg_get(cfg, ["kg", "cache_dir"], "data/cache/okvqa/slices")
    cache = SliceCache(cache_root)

    # slice config
    scfg = SliceConfig(
        hop_depth=int(_cfg_get(cfg, ["kg", "hop_depth"], 1)),
        top_k=int(_cfg_get(cfg, ["kg", "top_k"], 10)),
        relation_set=str(_cfg_get(cfg, ["kg", "relation_set"], "strict")),
        min_weight=float(_cfg_get(cfg, ["kg", "min_weight"], 0.0)),
        neighbor_limit=int(_cfg_get(cfg, ["kg", "neighbor_limit"], 200)),
        max_entities=int(_cfg_get(cfg, ["kg", "max_entities"], 6)),
        max_ngram=int(_cfg_get(cfg, ["kg", "max_ngram"], 3)),
        scorer_version=str(_cfg_get(cfg, ["kg", "scorer_version"], "v1")),
    )

    n = min(args.limit, len(ds))
    facts_per = []
    build_ms = []
    nonempty = 0
    cache_hits = 0

    samples_dir = Path(ctx.run_dir) / "slice_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_written = 0

    for i in tqdm(range(n), desc=f"build slices ({args.split})"):
        item = ds[i]
        qid = int(item["question_id"])
        iid = int(item["image_id"])
        qtext = str(item["question_text"])

        s, hit = build_slice(
            store=store,
            cache=cache,
            question_id=qid,
            image_id=iid,
            question_text=qtext,
            cfg=scfg,
        )
        if hit:
            cache_hits += 1

        nf = int(s["stats"]["n_facts"])
        facts_per.append(float(nf))
        build_ms.append(float(s["stats"]["build_ms"]))
        if nf > 0:
            nonempty += 1

        # write a small auditable sample set
        if samples_written < args.sample_count and nf > 0:
            md = []
            md.append(f"# Slice sample {samples_written+1}\n")
            md.append(f"**split:** {args.split}\n")
            md.append(f"**question_id:** {qid}  \\")
            md.append(f"**image_id:** {iid}\n")
            md.append(f"## Question\n{qtext}\n")
            md.append(f"## Entities\n- " + "\n- ".join(s.get("entities", [])) + "\n")
            md.append("## Top facts\n")
            for f in s.get("facts", []):
                md.append(f"- **{f['head']}** — *{f['relation']}* → **{f['tail']}**  (w={f['weight']:.3f}, score={f['score']:.3f}, hop={f['hop']})")
                if f.get("surface"):
                    md.append(f"  - surface: {f['surface']}")
            (samples_dir / f"{qid}_{iid}.md").write_text("\n".join(md), encoding="utf-8")
            samples_written += 1

    stats = {
        "status": "ok",
        "split": args.split,
        "n_examples": n,
        "nonempty_rate": nonempty / float(n) if n else 0.0,
        "cache_hit_rate": cache_hits / float(n) if n else 0.0,
        "facts_per_slice": summarize(facts_per),
        "build_ms": summarize(build_ms),
        "config": {
            "conceptnet_db": db_path,
            "cache_dir": str(cache_root),
            "slice_config": scfg.__dict__,
        },
        "samples_dir": str(samples_dir),
    }

    # write stats artifacts
    (Path(ctx.run_dir) / "slice_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    write_metrics(ctx.run_dir, stats)

    print("Run dir:", ctx.run_dir)
    print("slice_stats:", Path(ctx.run_dir) / "slice_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
