# src/train_fusion.py
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
from transformers import ViltProcessor

from run_metadata import init_run, write_metrics
from src.datasets.okvqa import OKVQADataset
from src.eval.normalize import normalize_answer
from src.eval.vqa_scoring import mean_vqa_soft_accuracy
from src.fusion.late_fusion import GatedFusion, WeightedAddFusion
from src.kg.cache import SliceCache
from src.kg.conceptnet_store import ConceptNetStore
from src.kg.knowledge_encoder import KnowledgeEncoder
from src.kg.slice_builder import SliceConfig, build_slice
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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
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


class FusionDataset(Dataset):
    def __init__(self, base: OKVQADataset, a2i: Dict[str, int], V: int, limit: int | None = None) -> None:
        self.base = base
        self.a2i = a2i
        self.V = V
        self.limit = limit

    def __len__(self) -> int:
        n = len(self.base)
        return min(n, self.limit) if self.limit is not None else n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.base[idx]
        return {
            "image": it["image"],
            "question_text": it["question_text"],
            "soft_target": make_soft_target(it["answers"], self.a2i, self.V),
            "gt_answers": it["answers"],
            "question_id": int(it["question_id"]),
            "image_id": int(it["image_id"]),
        }


def collate_fn(processor: ViltProcessor, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [x["image"] for x in batch]
    questions = [x["question_text"] for x in batch]
    enc = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True)
    targets = torch.stack([x["soft_target"] for x in batch])
    return {
        "inputs": enc,
        "targets": targets,
        "gt_answers": [x["gt_answers"] for x in batch],
        "question_ids": [x["question_id"] for x in batch],
        "image_ids": [x["image_id"] for x in batch],
        "question_texts": questions,
    }


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}
    targets = batch["targets"].to(device, non_blocking=True)
    return {**batch, "inputs": inputs, "targets": targets}


def apply_topn_rerank(base_logits: torch.Tensor, kg_logits: torch.Tensor, scale: torch.Tensor, topn: int) -> torch.Tensor:
    if topn is None or int(topn) <= 0:
        return base_logits + scale * kg_logits

    B, V = base_logits.shape
    topn = min(int(topn), V)

    idx = base_logits.topk(topn, dim=-1).indices  # [B,topn]
    base_top = base_logits.gather(1, idx)
    kg_top = kg_logits.gather(1, idx)

    fused = base_logits.clone()
    fused.scatter_(1, idx, base_top + scale * kg_top)
    return fused


@torch.no_grad()
def eval_loop(
    *,
    baseline: ViltForAnswerVocab,
    fusion,
    kg_enc: KnowledgeEncoder,
    store: ConceptNetStore,
    slice_cache: SliceCache,
    slice_cfg: SliceConfig,
    dl: DataLoader,
    idx_to_answer: List[str],
    device: str,
    use_kg: bool,
    topn_rerank: int,
) -> Dict[str, Any]:
    baseline.eval()
    fusion.eval()

    preds: List[str] = []
    gts: List[List[str]] = []
    loss_sum = 0.0
    n_batches = 0
    bce_sum = nn.BCEWithLogitsLoss(reduction="sum")

    for batch in tqdm(dl, desc=("eval+kg" if use_kg else "eval"), leave=False):
        b = move_batch_to_device(batch, device)
        base_logits = baseline(b["inputs"]).logits

        if use_kg:
            slices = []
            for qid, iid, qtxt in zip(b["question_ids"], b["image_ids"], b["question_texts"]):
                s, _ = build_slice(store=store, cache=slice_cache, question_id=int(qid), image_id=int(iid), question_text=str(qtxt), cfg=slice_cfg)
                slices.append(s)
            kg = kg_enc.encode_batch(slices)

            if isinstance(fusion, GatedFusion):
                gate = fusion.mlp(kg.kg_emb)  # [B,1]
                fused = apply_topn_rerank(base_logits, kg.kg_logits, gate, topn_rerank)
            else:
                a = fusion.alpha()  # scalar
                fused = apply_topn_rerank(base_logits, kg.kg_logits, a, topn_rerank)
        else:
            fused = base_logits

        loss = bce_sum(fused, b["targets"]) / fused.size(0)
        loss_sum += float(loss.item())
        n_batches += 1

        pred_idx = fused.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([idx_to_answer[i] for i in pred_idx])
        gts.extend(batch["gt_answers"])

    acc = mean_vqa_soft_accuracy(preds, gts)
    return {
        "val_vqa_soft_accuracy_mean": acc,
        "val_loss_mean": loss_sum / max(n_batches, 1),
        "n_eval_examples": len(preds),
        "kg_enabled": bool(use_kg),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="m5_fusion")
    args = ap.parse_args()

    cfg = load_config(args.config)

    out_root = _cfg_get(cfg, ["logging", "out_root"], "experiments/runs")
    ctx = init_run(tag=args.tag, config_path=args.config, out_root=out_root, include_pip_freeze=False, extra={"purpose": "fusion_training"})

    seed = int(_cfg_get(cfg, ["training", "seed"], 42))
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = _cfg_get(cfg, ["model", "answer_vocab", "path"], "data/processed/okvqa/answer_vocab.json")
    idx_to_answer, answer_to_idx = load_answer_vocab(vocab_path)
    V = len(idx_to_answer)

    baseline_ckpt = _cfg_get(cfg, ["baseline", "checkpoint_path"])
    backbone = _cfg_get(cfg, ["model", "backbone_checkpoint"], "dandelin/vilt-b32-mlm")
    processor = ViltProcessor.from_pretrained(backbone, use_fast=False)

    baseline = ViltForAnswerVocab(backbone_checkpoint=backbone, num_labels=V).to(device)
    if baseline_ckpt:
        state = torch.load(baseline_ckpt, map_location="cpu")
        baseline.load_state_dict(state["model_state_dict"])
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad_(False)

    store = ConceptNetStore(_cfg_get(cfg, ["conceptnet", "db_path"]))
    slice_cache = SliceCache(_cfg_get(cfg, ["kg", "cache_dir"], "data/cache/okvqa/slices"))
    slice_cfg = SliceConfig(
        hop_depth=int(_cfg_get(cfg, ["kg", "hop_depth"], 1)),
        top_k=int(_cfg_get(cfg, ["kg", "top_k"], 10)),
        relation_set=str(_cfg_get(cfg, ["kg", "relation_set"], "strict")),
        min_weight=float(_cfg_get(cfg, ["kg", "min_weight"], 0.0)),
        neighbor_limit=int(_cfg_get(cfg, ["kg", "neighbor_limit"], 200)),
        max_entities=int(_cfg_get(cfg, ["kg", "max_entities"], 6)),
        max_ngram=int(_cfg_get(cfg, ["kg", "max_ngram"], 3)),
        scorer_version=str(_cfg_get(cfg, ["kg", "scorer_version"], "v1")),
    )

    emb_model = _cfg_get(cfg, ["embed", "model_name"], "sentence-transformers/all-MiniLM-L6-v2")
    emb_cache_dir = _cfg_get(cfg, ["embed", "cache_dir"], "data/cache/embeddings")
    temperature = float(_cfg_get(cfg, ["embed", "temperature"], 10.0))

    kg_enc = KnowledgeEncoder(
        embedding_model=emb_model,
        answers=idx_to_answer,
        device=device,
        cache_dir=emb_cache_dir,
        temperature=temperature,
        fact_batch_size=int(_cfg_get(cfg, ["embed", "fact_batch_size"], 64)),
        answer_batch_size=int(_cfg_get(cfg, ["embed", "answer_batch_size"], 256)),
    )

    mode = str(_cfg_get(cfg, ["fusion", "mode"], "weighted")).lower()
    topn_rerank = int(_cfg_get(cfg, ["fusion", "topn_rerank"], 0))

    if mode == "gated":
        fusion = GatedFusion(emb_dim=int(kg_enc.answer_emb.shape[1]), hidden_dim=int(_cfg_get(cfg, ["fusion", "hidden_dim"], 128))).to(device)
    else:
        fusion = WeightedAddFusion(
            alpha_init=float(_cfg_get(cfg, ["fusion", "alpha_init"], 0.05)),
            learn_alpha=bool(_cfg_get(cfg, ["fusion", "learn_alpha"], True)),
        ).to(device)

    # data
    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])
    train_q = _cfg_get(cfg, ["data", "train_questions_json"])
    train_a = _cfg_get(cfg, ["data", "train_annotations_json"])
    val_q = _cfg_get(cfg, ["data", "val_questions_json"])
    val_a = _cfg_get(cfg, ["data", "val_annotations_json"])

    limit_train = int(_cfg_get(cfg, ["training", "limit_train"], 512))
    limit_val = int(_cfg_get(cfg, ["training", "limit_val"], 512))

    ds_train_base = OKVQADataset(f"{ann_dir}/{train_q}", f"{ann_dir}/{train_a}", img_root, load_images=True)
    ds_val_base = OKVQADataset(f"{ann_dir}/{val_q}", f"{ann_dir}/{val_a}", img_root, load_images=True)

    ds_train = FusionDataset(ds_train_base, answer_to_idx, V, limit=limit_train)
    ds_val = FusionDataset(ds_val_base, answer_to_idx, V, limit=limit_val)

    batch_size = int(_cfg_get(cfg, ["training", "batch_size"], 2))
    num_workers = int(_cfg_get(cfg, ["training", "num_workers"], 0))

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device=="cuda"),
                          collate_fn=lambda b: collate_fn(processor, b))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device=="cuda"),
                        collate_fn=lambda b: collate_fn(processor, b))

    epochs = int(_cfg_get(cfg, ["training", "epochs"], 1))
    lr = float(_cfg_get(cfg, ["training", "learning_rate"], 1e-3))
    wd = float(_cfg_get(cfg, ["training", "weight_decay"], 0.0))

    params = list(fusion.parameters())
    optimizer = None
    if params:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    bce_sum = nn.BCEWithLogitsLoss(reduction="sum")

    t0 = time.time()
    train_loss_sum = 0.0
    steps = 0

    for epoch in range(1, epochs + 1):
        fusion.train()
        if optimizer is None:
            break

        for batch in tqdm(dl_train, desc=f"train fusion {epoch}/{epochs}"):
            b = move_batch_to_device(batch, device)

            with torch.no_grad():
                base_logits = baseline(b["inputs"]).logits

            slices = []
            for qid, iid, qtxt in zip(b["question_ids"], b["image_ids"], b["question_texts"]):
                s, _ = build_slice(store=store, cache=slice_cache, question_id=int(qid), image_id=int(iid), question_text=str(qtxt), cfg=slice_cfg)
                slices.append(s)
            kg = kg_enc.encode_batch(slices)

            if isinstance(fusion, GatedFusion):
                gate = fusion.mlp(kg.kg_emb)
                fused = apply_topn_rerank(base_logits, kg.kg_logits, gate, topn_rerank)
            else:
                a = fusion.alpha()
                fused = apply_topn_rerank(base_logits, kg.kg_logits, a, topn_rerank)

            loss = bce_sum(fused, b["targets"]) / fused.size(0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            steps += 1

    val_base = eval_loop(baseline=baseline, fusion=fusion, kg_enc=kg_enc, store=store, slice_cache=slice_cache, slice_cfg=slice_cfg,
                         dl=dl_val, idx_to_answer=idx_to_answer, device=device, use_kg=False, topn_rerank=topn_rerank)
    val_kg = eval_loop(baseline=baseline, fusion=fusion, kg_enc=kg_enc, store=store, slice_cache=slice_cache, slice_cfg=slice_cfg,
                       dl=dl_val, idx_to_answer=idx_to_answer, device=device, use_kg=True, topn_rerank=topn_rerank)

    dt = time.time() - t0

    alpha_info: Dict[str, Any] = {}
    if isinstance(fusion, WeightedAddFusion):
        alpha_info = {"alpha": float(fusion.alpha().detach().cpu().item()), "learn_alpha": bool(fusion.learn_alpha)}

    metrics: Dict[str, Any] = {
        "status": "ok",
        "device": device,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_loss_mean": train_loss_sum / max(steps, 1),
        "seconds": dt,
        "fusion_mode": mode,
        "topn_rerank": topn_rerank,
        "alpha_info": alpha_info,
        "slice_config": slice_cfg.__dict__,
        "embed_model": emb_model,
        "temperature": temperature,
        "val_baseline": val_base,
        "val_fused": val_kg,
        "delta_val_acc": float(val_kg["val_vqa_soft_accuracy_mean"] - val_base["val_vqa_soft_accuracy_mean"]),
    }
    write_metrics(ctx.run_dir, metrics)

    ckpt_dir = Path(ctx.run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"fusion_state_dict": fusion.state_dict()}, ckpt_dir / "fusion.pt")

    print("Run dir:", ctx.run_dir)
    print("delta_val_acc:", metrics["delta_val_acc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
