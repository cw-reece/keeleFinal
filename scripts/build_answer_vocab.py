from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.datasets.okvqa import OKVQADataset, list_json_files
from src.eval.normalize import normalize_answer
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
    ap.add_argument("--top_n", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    ann_dir = Path(_cfg_get(cfg, ["data", "annotations_dir"], "data/raw/okvqa/annotations"))
    img_root = Path(_cfg_get(cfg, ["data", "coco_images_root"], "data/raw/okvqa/images"))

    train_q = _cfg_get(cfg, ["data", "train_questions_json"])
    train_a = _cfg_get(cfg, ["data", "train_annotations_json"])

    if not ann_dir.exists():
        print(f"ERROR: annotations_dir not found: {ann_dir}")
        return 2

    if not all([train_q, train_a]):
        print("Config missing one or more required fields:")
        print("  data.train_questions_json")
        print("  data.train_annotations_json")
        print("\nFound JSON files in annotations_dir:")
        for f in list_json_files(ann_dir):
            print(" -", f)
        return 3

    train_qp = ann_dir / train_q
    train_ap = ann_dir / train_a
    for p in [train_qp, train_ap]:
        if not p.exists():
            print(f"ERROR: missing file: {p}")
            print("Available JSON files:")
            for f in list_json_files(ann_dir):
                print(" -", f)
            return 4

    top_n = args.top_n or int(_cfg_get(cfg, ["model", "answer_vocab", "top_n"], 3000))
    out_path = Path(_cfg_get(cfg, ["model", "answer_vocab", "path"], "data/processed/okvqa/answer_vocab.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_train = OKVQADataset(train_qp, train_ap, img_root, load_images=False)

    c = Counter()
    for r in ds_train.records:
        for a in r.answers:
            na = normalize_answer(a)
            if na:
                c[na] += 1

    most = c.most_common(top_n)
    vocab = {
        "top_n": top_n,
        "size": len(most),
        "items": [{"answer": a, "count": int(cnt)} for a, cnt in most],
    }

    out_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    print(f"Wrote vocab: {out_path} (size={len(most)})")
    print("Top 10:", [a for a, _ in most[:10]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

