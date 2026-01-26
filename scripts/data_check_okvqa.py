from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.okvqa import OKVQADataset, list_json_files
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
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max_image_checks", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)

    ann_dir = Path(_cfg_get(cfg, ["data", "annotations_dir"], "data/raw/okvqa/annotations"))
    img_root = Path(_cfg_get(cfg, ["data", "coco_images_root"], "data/raw/okvqa/images"))

    train_q = _cfg_get(cfg, ["data", "train_questions_json"])
    train_a = _cfg_get(cfg, ["data", "train_annotations_json"])
    val_q = _cfg_get(cfg, ["data", "val_questions_json"])
    val_a = _cfg_get(cfg, ["data", "val_annotations_json"])

    if not ann_dir.exists():
        print(f"ERROR: annotations_dir not found: {ann_dir}")
        return 2

    # If you haven’t set filenames yet, this prints what’s available.
    if not all([train_q, train_a, val_q, val_a]):
        print("Config missing one or more required fields:")
        print("  data.train_questions_json")
        print("  data.train_annotations_json")
        print("  data.val_questions_json")
        print("  data.val_annotations_json")
        print("\nFound JSON files in annotations_dir:")
        for f in list_json_files(ann_dir):
            print(" -", f)
        return 3

    train_qp = ann_dir / train_q
    train_ap = ann_dir / train_a
    val_qp = ann_dir / val_q
    val_ap = ann_dir / val_a

    for p in [train_qp, train_ap, val_qp, val_ap]:
        if not p.exists():
            print(f"ERROR: missing file: {p}")
            print("Available JSON files:")
            for f in list_json_files(ann_dir):
                print(" -", f)
            return 4

    ds_train = OKVQADataset(train_qp, train_ap, img_root, load_images=False)
    ds_val = OKVQADataset(val_qp, val_ap, img_root, load_images=False)

    print("=== OK-VQA integrity check ===")
    print(f"annotations_dir: {ann_dir}")
    print(f"coco_images_root: {img_root}")
    print(f"train examples: {len(ds_train)}")
    print(f"val examples:   {len(ds_val)}")

    def image_exists_rate(ds: OKVQADataset) -> float:
        n = min(len(ds), args.max_image_checks)
        found = 0
        for i in range(n):
            item = ds[i]
            if item.get("image_path"):
                found += 1
        return found / float(n) if n > 0 else 0.0

    train_rate = image_exists_rate(ds_train)
    val_rate = image_exists_rate(ds_val)
    print(f"image exists rate (train, first {min(len(ds_train), args.max_image_checks)}): {train_rate:.3f}")
    print(f"image exists rate (val,   first {min(len(ds_val), args.max_image_checks)}): {val_rate:.3f}")

    print("\n--- samples (train) ---")
    for i in range(min(args.samples, len(ds_train))):
        it = ds_train[i]
        print(f"[{i}] image_id={it['image_id']} question_id={it['question_id']}")
        print("Q:", it["question_text"])
        print("A:", it["answers"][:5])
        print("img_path:", it.get("image_path"))
        print()

    print("--- samples (val) ---")
    for i in range(min(args.samples, len(ds_val))):
        it = ds_val[i]
        print(f"[{i}] image_id={it['image_id']} question_id={it['question_id']}")
        print("Q:", it["question_text"])
        print("A:", it["answers"][:5])
        print("img_path:", it.get("image_path"))
        print()

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
