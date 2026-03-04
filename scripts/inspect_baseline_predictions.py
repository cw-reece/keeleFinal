# scripts/inspect_baseline_predictions.py
"""Inspect baseline predictions distribution to catch 'always predicts X' failures.

Usage:
  python -m scripts.inspect_baseline_predictions \
    --run_dir experiments/runs/<run_id> \
    --config configs/baseline_train.yaml \
    --split val --limit 500
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViltProcessor

from src.datasets.okvqa import OKVQADataset
from src.eval.vqa_scoring import mean_vqa_soft_accuracy
from src.models.vilt_classifier import ViltForAnswerVocab
from src.utils.config import load_config


def _cfg_get(cfg: dict, path: list[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_vocab(vocab_path: str | Path) -> list[str]:
    data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    return [str(x["answer"]) for x in data.get("items", [])]


def collate_fn(processor: ViltProcessor, batch):
    images = [x["image"] for x in batch]
    questions = [x["question_text"] for x in batch]
    enc = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True)
    return {"inputs": enc, "gt_answers": [x["answers"] for x in batch]}


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config(args.config)

    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])

    q_file = _cfg_get(cfg, ["data", "val_questions_json"] if args.split == "val" else ["data", "train_questions_json"])
    a_file = _cfg_get(cfg, ["data", "val_annotations_json"] if args.split == "val" else ["data", "train_annotations_json"])

    vocab_path = _cfg_get(cfg, ["model", "answer_vocab", "path"], "data/processed/okvqa/answer_vocab.json")
    idx_to_answer = load_vocab(vocab_path)

    ckpt = Path(args.run_dir) / "checkpoints" / "model.pt"
    state = torch.load(ckpt, map_location="cpu")

    backbone = state.get("backbone_checkpoint", _cfg_get(cfg, ["model", "backbone_checkpoint"], "dandelin/vilt-b32-mlm"))
    processor = ViltProcessor.from_pretrained(backbone, use_fast=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViltForAnswerVocab(backbone_checkpoint=backbone, num_labels=len(idx_to_answer)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    ds_base = OKVQADataset(f"{ann_dir}/{q_file}", f"{ann_dir}/{a_file}", img_root, load_images=True)

    class Wrap(torch.utils.data.Dataset):
        def __len__(self): return min(args.limit, len(ds_base))
        def __getitem__(self, i):
            it = ds_base[i]
            return {"image": it["image"], "question_text": it["question_text"], "answers": it["answers"]}

    dl = DataLoader(Wrap(), batch_size=8, shuffle=False, num_workers=0, collate_fn=lambda b: collate_fn(processor, b))

    counts = Counter()
    maxlogits = []
    preds = []
    gts = []

    for batch in tqdm(dl, desc="inspect"):
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        out = model(inputs)
        logits = out.logits
        mx, idx = logits.max(dim=-1)
        maxlogits.extend(mx.detach().cpu().tolist())
        idx = idx.detach().cpu().tolist()
        pred_answers = [idx_to_answer[i] for i in idx]
        preds.extend(pred_answers)
        gts.extend(batch["gt_answers"])
        counts.update(pred_answers)

    acc = mean_vqa_soft_accuracy(preds, gts)

    print("Top predicted answers:")
    for a, c in counts.most_common(10):
        print(f"  {a}: {c}")

    if maxlogits:
        import statistics
        print("max-logit mean:", statistics.mean(maxlogits))
        print("max-logit max :", max(maxlogits))

    print("subset mean VQA soft accuracy:", acc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
