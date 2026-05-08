# scripts/eval_baseline_ckpt.py
"""Evaluate a baseline checkpoint on OK-VQA split.

Usage:
  python -m scripts.eval_baseline_ckpt --config configs/baseline_train_v4_suggested.yaml \
    --checkpoint experiments/runs/BASELINE_FREEZE_20260312_1456/checkpoints/model.pt \
    --split val --limit 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViltProcessor

from src.datasets.okvqa import OKVQADataset
from src.eval.normalize import normalize_answer
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


def load_answer_vocab(path: str | Path) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items", [])
    return [normalize_answer(str(x["answer"])) for x in items]


class QAOnlyDataset(Dataset):
    def __init__(self, base: OKVQADataset, limit: Optional[int] = None) -> None:
        self.base = base
        self.limit = limit

    def __len__(self) -> int:
        n = len(self.base)
        return n if not self.limit or self.limit <= 0 else min(n, self.limit)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.base[idx]
        return {"image": it["image"], "question_text": it["question_text"], "answers": it["answers"]}


def collate_fn(processor: ViltProcessor, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [x["image"] for x in batch]
    questions = [x["question_text"] for x in batch]
    enc = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True)
    return {"inputs": enc, "answers": [x["answers"] for x in batch]}


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)

    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])
    train_q = _cfg_get(cfg, ["data", "train_questions_json"])
    train_a = _cfg_get(cfg, ["data", "train_annotations_json"])
    val_q = _cfg_get(cfg, ["data", "val_questions_json"])
    val_a = _cfg_get(cfg, ["data", "val_annotations_json"])
    q, a = (train_q, train_a) if args.split == "train" else (val_q, val_a)

    ds_base = OKVQADataset(f"{ann_dir}/{q}", f"{ann_dir}/{a}", img_root, load_images=True)
    ds = QAOnlyDataset(ds_base, limit=args.limit if args.limit > 0 else None)

    vocab_path = _cfg_get(cfg, ["model", "answer_vocab", "path"], "data/processed/okvqa/answer_vocab.json")
    idx_to_answer = load_answer_vocab(vocab_path)
    V = len(idx_to_answer)

    backbone = _cfg_get(cfg, ["model", "backbone_checkpoint"], "dandelin/vilt-b32-mlm")
    processor = ViltProcessor.from_pretrained(backbone, use_fast=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViltForAnswerVocab(backbone_checkpoint=backbone, num_labels=V).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device=="cuda"),
                    collate_fn=lambda b: collate_fn(processor, b))

    preds: List[str] = []
    gts: List[List[str]] = []

    for batch in tqdm(dl, desc=f"eval_baseline_{args.split}"):
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        logits = model(inputs).logits
        pred_idx = logits.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([idx_to_answer[i] for i in pred_idx])
        gts.extend(batch["answers"])

    acc = mean_vqa_soft_accuracy(preds, gts)
    print(f"split={args.split} n={len(preds)} vqa_soft_acc={acc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
