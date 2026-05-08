# scripts/eval_fusion_run.py
"""Eval baseline vs fused for a given fusion run_dir.

Usage:
  python -m scripts.eval_fusion_run --config configs/fusion_train_v3_topn.yaml \
    --fusion_run_dir experiments/runs/20260313_145031_m5_topn20_weighted_fullval \
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
        return {
            "image": it["image"],
            "question_text": it["question_text"],
            "answers": it["answers"],
            "question_id": int(it["question_id"]),
            "image_id": int(it["image_id"]),
        }


def collate_fn(processor: ViltProcessor, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [x["image"] for x in batch]
    questions = [x["question_text"] for x in batch]
    enc = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True)
    return {
        "inputs": enc,
        "answers": [x["answers"] for x in batch],
        "question_ids": [x["question_id"] for x in batch],
        "image_ids": [x["image_id"] for x in batch],
        "question_texts": questions,
    }


def apply_topn_rerank(base_logits: torch.Tensor, kg_logits: torch.Tensor, scale: torch.Tensor, topn: int) -> torch.Tensor:
    if topn is None or int(topn) <= 0:
        return base_logits + scale * kg_logits
    B, V = base_logits.shape
    topn = min(int(topn), V)
    idx = base_logits.topk(topn, dim=-1).indices
    base_top = base_logits.gather(1, idx)
    kg_top = kg_logits.gather(1, idx)
    fused = base_logits.clone()
    fused.scatter_(1, idx, base_top + scale * kg_top)
    return fused


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fusion_run_dir", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.fusion_run_dir)
    run_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    fusion_mode = str(run_metrics.get("fusion_mode", "weighted")).lower()
    topn = int(run_metrics.get("topn_rerank") or 0)
    temperature = float(run_metrics.get("temperature") or _cfg_get(cfg, ["embed", "temperature"], 2.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = _cfg_get(cfg, ["model", "answer_vocab", "path"], "data/processed/okvqa/answer_vocab.json")
    idx_to_answer = load_answer_vocab(vocab_path)
    V = len(idx_to_answer)

    backbone = _cfg_get(cfg, ["model", "backbone_checkpoint"], "dandelin/vilt-b32-mlm")
    processor = ViltProcessor.from_pretrained(backbone, use_fast=False)

    baseline_ckpt = _cfg_get(cfg, ["baseline", "checkpoint_path"])
    baseline = ViltForAnswerVocab(backbone_checkpoint=backbone, num_labels=V).to(device)
    state = torch.load(baseline_ckpt, map_location="cpu")
    baseline.load_state_dict(state["model_state_dict"], strict=True)
    baseline.eval()

    store = ConceptNetStore(_cfg_get(cfg, ["conceptnet", "db_path"]))
    slice_cache = SliceCache(_cfg_get(cfg, ["kg", "cache_dir"], "data/cache/okvqa/slices"))
    scfg = run_metrics.get("slice_config") or {}
    slice_cfg = SliceConfig(
        hop_depth=int(scfg.get("hop_depth", 1)),
        top_k=int(scfg.get("top_k", 10)),
        relation_set=str(scfg.get("relation_set", "strict")),
        min_weight=float(scfg.get("min_weight", 0.0)),
        neighbor_limit=int(scfg.get("neighbor_limit", 200)),
        max_entities=int(scfg.get("max_entities", 6)),
        max_ngram=int(scfg.get("max_ngram", 3)),
        scorer_version=str(scfg.get("scorer_version", "v1")),
    )

    emb_model = str(run_metrics.get("embed_model") or _cfg_get(cfg, ["embed", "model_name"], "sentence-transformers/all-MiniLM-L6-v2"))
    emb_cache_dir = _cfg_get(cfg, ["embed", "cache_dir"], "data/cache/embeddings")
    kg_enc = KnowledgeEncoder(
        embedding_model=emb_model,
        answers=idx_to_answer,
        device=device,
        cache_dir=emb_cache_dir,
        temperature=temperature,
        fact_batch_size=int(_cfg_get(cfg, ["embed", "fact_batch_size"], 128)),
        answer_batch_size=int(_cfg_get(cfg, ["embed", "answer_batch_size"], 512)),
    )

    fpt = run_dir / "checkpoints" / "fusion.pt"
    fstate = torch.load(fpt, map_location="cpu")["fusion_state_dict"]
    if fusion_mode == "gated":
        fusion = GatedFusion(emb_dim=int(kg_enc.answer_emb.shape[1]), hidden_dim=128).to(device)
    else:
        fusion = WeightedAddFusion(alpha_init=0.05, learn_alpha=True).to(device)
    fusion.load_state_dict(fstate, strict=True)
    fusion.eval()

    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])
    if args.split == "train":
        q = _cfg_get(cfg, ["data", "train_questions_json"])
        a = _cfg_get(cfg, ["data", "train_annotations_json"])
    else:
        q = _cfg_get(cfg, ["data", "val_questions_json"])
        a = _cfg_get(cfg, ["data", "val_annotations_json"])

    ds_base = OKVQADataset(f"{ann_dir}/{q}", f"{ann_dir}/{a}", img_root, load_images=True)
    ds = QAOnlyDataset(ds_base, limit=args.limit if args.limit > 0 else None)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device=="cuda"),
                    collate_fn=lambda b: collate_fn(processor, b))

    preds_base: List[str] = []
    preds_fused: List[str] = []
    gts: List[List[str]] = []

    for batch in tqdm(dl, desc=f"eval_fusion_{args.split}"):
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        base_logits = baseline(inputs).logits

        slices = []
        for qid, iid, qtxt in zip(batch["question_ids"], batch["image_ids"], batch["question_texts"]):
            s, _ = build_slice(store=store, cache=slice_cache, question_id=int(qid), image_id=int(iid), question_text=str(qtxt), cfg=slice_cfg)
            slices.append(s)
        kg = kg_enc.encode_batch(slices)

        if fusion_mode == "gated":
            gate = fusion.mlp(kg.kg_emb)
            fused_logits = apply_topn_rerank(base_logits, kg.kg_logits, gate, topn)
        else:
            a_scale = fusion.alpha()
            fused_logits = apply_topn_rerank(base_logits, kg.kg_logits, a_scale, topn)

        idx_b = base_logits.argmax(dim=-1).detach().cpu().tolist()
        idx_f = fused_logits.argmax(dim=-1).detach().cpu().tolist()
        preds_base.extend([idx_to_answer[i] for i in idx_b])
        preds_fused.extend([idx_to_answer[i] for i in idx_f])
        gts.extend(batch["answers"])

    acc_b = mean_vqa_soft_accuracy(preds_base, gts)
    acc_f = mean_vqa_soft_accuracy(preds_fused, gts)
    print(f"split={args.split} n={len(gts)} baseline_acc={acc_b} fused_acc={acc_f} delta={acc_f-acc_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
