# scripts/vocab_coverage.py
"""Measure answer-vocabulary coverage on OK-VQA splits.

Why:
Your classifier can only predict answers that exist in your answer vocab.
If coverage is low, accuracy will be capped no matter how good the model is.

Usage:
  python -m scripts.vocab_coverage --config configs/baseline.yaml --vocab data/processed/okvqa/answer_vocab.json

Outputs:
  Prints % of examples with at least one in-vocab answer (train and val).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.datasets.okvqa import OKVQADataset
from src.eval.normalize import normalize_answer
from src.utils.config import load_config


def _cfg_get(cfg: dict, path: list[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_vocab(vocab_path: str | Path) -> set[str]:
    data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    items = data.get("items", [])
    return set(str(x["answer"]) for x in items)


def split_coverage(ds: OKVQADataset, vocab: set[str]) -> float:
    n = len(ds)
    hit = 0
    for r in ds.records:
        ok = False
        for a in r.answers:
            na = normalize_answer(a)
            if na in vocab:
                ok = True
                break
        if ok:
            hit += 1
    return hit / float(n) if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--vocab", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])

    train_q = _cfg_get(cfg, ["data", "train_questions_json"])
    train_a = _cfg_get(cfg, ["data", "train_annotations_json"])
    val_q = _cfg_get(cfg, ["data", "val_questions_json"])
    val_a = _cfg_get(cfg, ["data", "val_annotations_json"])

    vocab = load_vocab(args.vocab)

    ds_train = OKVQADataset(f"{ann_dir}/{train_q}", f"{ann_dir}/{train_a}", img_root, load_images=False)
    ds_val = OKVQADataset(f"{ann_dir}/{val_q}", f"{ann_dir}/{val_a}", img_root, load_images=False)

    tr = split_coverage(ds_train, vocab)
    va = split_coverage(ds_val, vocab)

    print("=== Vocab coverage ===")
    print("vocab size:", len(vocab))
    print(f"train coverage: {tr:.3f} ({tr*100:.1f}%)")
    print(f"val coverage:   {va:.3f} ({va*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
