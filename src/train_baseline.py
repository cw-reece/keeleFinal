# src/train_baseline.py
"""Baseline training for OK-VQA (Milestone 2) — VQA-init + class-imbalance fix.

This version addresses two common failure modes that produce near-zero accuracy even with good vocab coverage:

1) Initialization: starting from MLM-only ViLT weights can collapse.
   Fix: optionally initialize the encoder from a VQA-finetuned checkpoint
   (dandelin/vilt-b32-finetuned-vqa) while keeping a fresh vocab head.

2) Extreme class imbalance: targets are sparse (few positives among 10k labels).
   Fix: use BCEWithLogitsLoss(reduction='sum')/batch AND apply a reasonable pos_weight.

Config additions (optional):
model:
  init_from_vqa_checkpoint: "dandelin/vilt-b32-finetuned-vqa"

training:
  pos_weight: 50                # override auto (recommended range: 10..200)
  pos_weight_auto_samples: 512  # how many train samples to estimate avg positives
  pos_weight_max: 200           # clamp auto pos_weight

Usage:
  python -m src.train_baseline --config configs/baseline_train_v4_suggested.yaml --tag baseline_v4_vocab10k_ep3_fullval
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViltProcessor, ViltForQuestionAnswering

from run_metadata import init_run, write_metrics
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


def load_answer_vocab(path: str | Path) -> Tuple[List[str], Dict[str, int]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("items", [])
    answers = [str(x["answer"]) for x in items]
    a2i = {a: i for i, a in enumerate(answers)}
    return answers, a2i


def make_soft_target(answers: List[str], a2i: Dict[str, int], num_labels: int) -> torch.Tensor:
    c = Counter()
    for a in answers:
        na = normalize_answer(a)
        if na:
            c[na] += 1

    y = torch.zeros(num_labels, dtype=torch.float32)
    for na, cnt in c.items():
        idx = a2i.get(na)
        if idx is None:
            continue
        y[idx] = min(cnt / 3.0, 1.0)
    return y


def estimate_avg_positives(ds_base: OKVQADataset, a2i: Dict[str, int], samples: int = 512) -> float:
    """Estimate avg number of in-vocab positive labels per example without building full vectors."""
    n = min(samples, len(ds_base.records))
    tot = 0
    for i in range(n):
        r = ds_base.records[i]
        pos = set()
        for a in r.answers:
            na = normalize_answer(a)
            if na in a2i:
                pos.add(na)
        tot += len(pos)
    return (tot / float(n)) if n > 0 else 1.0


class OKVQABaselineDataset(Dataset):
    def __init__(self, base: OKVQADataset, answer_to_idx: Dict[str, int], num_labels: int, limit: int | None = None) -> None:
        self.base = base
        self.answer_to_idx = answer_to_idx
        self.num_labels = num_labels
        self.limit = limit

    def __len__(self) -> int:
        n = len(self.base)
        return min(n, self.limit) if self.limit is not None else n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        return {
            "image": item["image"],
            "question_text": item["question_text"],
            "soft_target": make_soft_target(item["answers"], self.answer_to_idx, self.num_labels),
            "gt_answers": item["answers"],
        }


def collate_fn(processor: ViltProcessor, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [x["image"] for x in batch]
    questions = [x["question_text"] for x in batch]
    enc = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True)
    targets = torch.stack([x["soft_target"] for x in batch])
    gt_answers = [x["gt_answers"] for x in batch]
    return {"inputs": enc, "targets": targets, "gt_answers": gt_answers}


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}
    targets = batch["targets"].to(device, non_blocking=True)
    return {"inputs": inputs, "targets": targets, "gt_answers": batch["gt_answers"]}


@torch.no_grad()
def evaluate(model: ViltForAnswerVocab, dataloader: DataLoader, idx_to_answer: List[str], device: str, bce_sum) -> Dict[str, Any]:
    model.eval()
    preds: List[str] = []
    gts: List[List[str]] = []
    loss_sum = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="eval", leave=False):
        b = move_batch_to_device(batch, device)
        out = model(b["inputs"])
        logits = out.logits
        targets = b["targets"]

        loss = bce_sum(logits, targets) / logits.size(0)
        loss_sum += float(loss.item())
        n_batches += 1

        pred_idx = logits.argmax(dim=-1).tolist()
        preds.extend([idx_to_answer[i] for i in pred_idx])
        gts.extend(batch["gt_answers"])

    acc = mean_vqa_soft_accuracy(preds, gts)
    return {
        "val_vqa_soft_accuracy_mean": acc,
        "val_loss_mean": (loss_sum / max(n_batches, 1)),
        "n_eval_examples": len(preds),
    }


def maybe_init_from_vqa(model: ViltForAnswerVocab, vqa_ckpt: str) -> Dict[str, Any]:
    """Initialize the ViLT encoder weights from a VQA-finetuned checkpoint."""
    info: Dict[str, Any] = {"enabled": False}
    if not vqa_ckpt:
        return info

    donor = ViltForQuestionAnswering.from_pretrained(vqa_ckpt)
    incompatible = model.vilt.load_state_dict(donor.vilt.state_dict(), strict=False)
    # In newer torch/transformers, load_state_dict returns an IncompatibleKeys object.
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    info = {
        "enabled": True,
        "vqa_checkpoint": vqa_ckpt,
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
    }
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="baseline_train")
    args = ap.parse_args()

    cfg = load_config(args.config)

    out_root = _cfg_get(cfg, ["logging", "out_root"], "experiments/runs")
    ctx = init_run(
        tag=args.tag,
        config_path=args.config,
        out_root=out_root,
        include_pip_freeze=bool(_cfg_get(cfg, ["logging", "include_pip_freeze"], False)),
        extra={"purpose": "baseline_training"},
    )

    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])
    train_q = _cfg_get(cfg, ["data", "train_questions_json"])
    train_a = _cfg_get(cfg, ["data", "train_annotations_json"])
    val_q = _cfg_get(cfg, ["data", "val_questions_json"])
    val_a = _cfg_get(cfg, ["data", "val_annotations_json"])
    if not all([ann_dir, img_root, train_q, train_a, val_q, val_a]):
        write_metrics(ctx.run_dir, {"status": "error", "error": "Missing data paths in config."})
        return 2

    seed = int(_cfg_get(cfg, ["training", "seed"], 42))
    epochs = int(_cfg_get(cfg, ["training", "epochs"], 3))
    batch_size = int(_cfg_get(cfg, ["training", "batch_size"], 4))
    lr = float(_cfg_get(cfg, ["training", "learning_rate"], 5e-5))
    weight_decay = float(_cfg_get(cfg, ["training", "weight_decay"], 0.01))
    mixed_precision = bool(_cfg_get(cfg, ["training", "mixed_precision"], True))
    limit_train = _cfg_get(cfg, ["training", "limit_train"], None)
    limit_val = _cfg_get(cfg, ["training", "limit_val"], None)
    num_workers = int(_cfg_get(cfg, ["training", "num_workers"], 0))

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = _cfg_get(cfg, ["model", "answer_vocab", "path"], "data/processed/okvqa/answer_vocab.json")
    idx_to_answer, answer_to_idx = load_answer_vocab(vocab_path)
    num_labels = len(idx_to_answer)

    backbone = _cfg_get(cfg, ["model", "backbone_checkpoint"], "dandelin/vilt-b32-mlm")
    processor = ViltProcessor.from_pretrained(backbone, use_fast=False)
    model = ViltForAnswerVocab(backbone_checkpoint=backbone, num_labels=num_labels, dropout=float(_cfg_get(cfg, ["model", "dropout"], 0.1))).to(device)

    # Optional: initialize encoder from VQA checkpoint
    vqa_init_ckpt = _cfg_get(cfg, ["model", "init_from_vqa_checkpoint"], "")
    vqa_init_info = maybe_init_from_vqa(model, vqa_init_ckpt)

    ds_train_base = OKVQADataset(f"{ann_dir}/{train_q}", f"{ann_dir}/{train_a}", img_root, load_images=True)
    ds_val_base = OKVQADataset(f"{ann_dir}/{val_q}", f"{ann_dir}/{val_a}", img_root, load_images=True)

    if limit_train is not None:
        limit_train = int(limit_train)
    if limit_val is not None:
        limit_val = int(limit_val)

    # pos_weight
    pos_weight_cfg = _cfg_get(cfg, ["training", "pos_weight"], None)
    pos_weight_max = float(_cfg_get(cfg, ["training", "pos_weight_max"], 200.0))
    pos_weight_auto_samples = int(_cfg_get(cfg, ["training", "pos_weight_auto_samples"], 512))

    if pos_weight_cfg is None:
        avg_pos = estimate_avg_positives(ds_train_base, answer_to_idx, samples=pos_weight_auto_samples)
        if avg_pos <= 0:
            avg_pos = 1.0
        pos_weight_scalar = min(max((num_labels - avg_pos) / avg_pos, 1.0), pos_weight_max)
        pos_weight_source = "auto"
        avg_pos_est = avg_pos
    else:
        pos_weight_scalar = float(pos_weight_cfg)
        pos_weight_scalar = min(max(pos_weight_scalar, 1.0), pos_weight_max)
        pos_weight_source = "config"
        avg_pos_est = None

    pos_weight_tensor = torch.full((num_labels,), pos_weight_scalar, device=device)

    ds_train = OKVQABaselineDataset(ds_train_base, answer_to_idx, num_labels, limit=limit_train)
    ds_val = OKVQABaselineDataset(ds_val_base, answer_to_idx, num_labels, limit=limit_val)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_fn(processor, b),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_fn(processor, b),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    bce_sum = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight_tensor)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision and device == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device == "cuda"))

    t0 = time.time()
    train_loss_sum = 0.0
    n_steps = 0
    val_metrics: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}"):
            b = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.autocast(device_type="cuda", enabled=True):
                    out = model(b["inputs"])
                    loss = bce_sum(out.logits, b["targets"]) / out.logits.size(0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(b["inputs"])
                loss = bce_sum(out.logits, b["targets"]) / out.logits.size(0)
                loss.backward()
                optimizer.step()

            train_loss_sum += float(loss.item())
            n_steps += 1

        val_metrics = evaluate(model, val_loader, idx_to_answer, device, bce_sum)

    dt = time.time() - t0
    train_loss_mean = train_loss_sum / max(n_steps, 1)

    ckpt_dir = Path(ctx.run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone_checkpoint": backbone,
            "idx_to_answer": idx_to_answer,
            "seed": seed,
            "pos_weight": pos_weight_scalar,
            "vqa_init": vqa_init_info,
        },
        ckpt_path,
    )

    metrics: Dict[str, Any] = {
        "status": "ok",
        "device": device,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "mixed_precision": mixed_precision,
        "limit_train": limit_train,
        "limit_val": limit_val,
        "num_workers": num_workers,
        "num_labels": num_labels,
        "train_loss_mean": train_loss_mean,
        "seconds": dt,
        "checkpoint_path": str(ckpt_path),
        "loss_reduction": "sum_div_batch",
        "pos_weight": pos_weight_scalar,
        "pos_weight_source": pos_weight_source,
        "avg_pos_estimate": avg_pos_est,
        "vqa_init": vqa_init_info,
    }
    metrics.update(val_metrics)

    write_metrics(ctx.run_dir, metrics)

    print("Run dir:", ctx.run_dir)
    print("Val VQA soft accuracy:", metrics.get("val_vqa_soft_accuracy_mean"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
