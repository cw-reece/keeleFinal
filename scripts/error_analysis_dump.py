# scripts/error_analysis_dump.py
"""Dump per-example baseline vs fused predictions + KG slice facts for error analysis.

This is designed for *manual* qualitative analysis: pick ~6-12 interesting cases and write
about why KG helped/hurt/was ignored.

Examples:
  # Baseline vs Top-N weighted fusion run
  python -m scripts.error_analysis_dump \
    --config configs/fusion_train_v3_topn.yaml \
    --fusion_run_dir experiments/runs/20260313_145031_m5_topn20_weighted_fullval \
    --split val --limit 300 \
    --out_dir reports/error_analysis/m5_topn20_weighted_val300

  # Baseline vs Top-N gated fusion run
  python -m scripts.error_analysis_dump \
    --config configs/fusion_train_v3_topn.yaml \
    --fusion_run_dir experiments/runs/20260313_145705_m5_topn20_gated_fullval \
    --split val --limit 300 \
    --out_dir reports/error_analysis/m5_topn20_gated_val300

Outputs:
  - predictions.jsonl   (one JSON object per example)
  - summary.json        (counts + quick stats)
  - selected_cases.md   (ready-to-paste qualitative examples)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViltProcessor

from src.datasets.okvqa import OKVQADataset
from src.eval.normalize import normalize_answer
from src.kg.cache import SliceCache
from src.kg.conceptnet_store import ConceptNetStore
from src.kg.knowledge_encoder import KnowledgeEncoder
from src.kg.slice_builder import SliceConfig, build_slice
from src.models.vilt_classifier import ViltForAnswerVocab
from src.utils.config import load_config

from src.fusion.late_fusion import GatedFusion, WeightedAddFusion


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


def vqa_soft_score(pred: str, gt_answers: List[str]) -> float:
    """VQA-style soft score: min(count(pred)/3, 1)."""
    p = normalize_answer(pred)
    c = Counter(normalize_answer(a) for a in gt_answers)
    return min(c.get(p, 0) / 3.0, 1.0)


def topk_list(logits: torch.Tensor, idx_to_answer: List[str], k: int = 5) -> List[Dict[str, Any]]:
    vals, idx = torch.topk(logits, k=min(k, logits.numel()))
    vals = vals.detach().cpu().tolist()
    idx = idx.detach().cpu().tolist()
    out = []
    for i, v in zip(idx, vals):
        out.append({"answer": idx_to_answer[i], "logit": float(v), "idx": int(i)})
    return out


class QAOnlyDataset(Dataset):
    def __init__(self, base: OKVQADataset, limit: Optional[int] = None) -> None:
        self.base = base
        self.limit = limit

    def __len__(self) -> int:
        n = len(self.base)
        return min(n, self.limit) if self.limit is not None else n

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


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    enc = {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}
    return {**batch, "inputs": enc}


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


def facts_preview(slice_obj: Dict[str, Any], max_facts: int = 10) -> List[Dict[str, Any]]:
    facts = slice_obj.get("facts", []) or []
    out = []
    for f in facts[:max_facts]:
        out.append({
            "head": f.get("head"),
            "relation": f.get("relation"),
            "tail": f.get("tail"),
            "score": f.get("score"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Fusion config (kg + embed + baseline ckpt paths).")
    ap.add_argument("--fusion_run_dir", required=True, help="Run dir containing checkpoints/fusion.pt and metrics.json")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5, help="Top-K answers to store per example for baseline/fused.")
    ap.add_argument("--max_facts", type=int, default=10, help="How many facts to store per example.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.fusion_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics.json in {run_dir}")
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    fusion_mode = str(run_metrics.get("fusion_mode", "weighted")).lower()
    topn = int(run_metrics.get("topn_rerank") or 0)
    temperature = float(run_metrics.get("temperature") or _cfg_get(cfg, ["embed", "temperature"], 2.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load vocab / model / processor
    vocab_path = _cfg_get(cfg, ["model", "answer_vocab", "path"], "data/processed/okvqa/answer_vocab.json")
    idx_to_answer = load_answer_vocab(vocab_path)
    V = len(idx_to_answer)

    backbone = _cfg_get(cfg, ["model", "backbone_checkpoint"], "dandelin/vilt-b32-mlm")
    processor = ViltProcessor.from_pretrained(backbone, use_fast=False)

    baseline_ckpt = _cfg_get(cfg, ["baseline", "checkpoint_path"])
    if not baseline_ckpt:
        raise SystemExit("baseline.checkpoint_path missing in config")
    baseline = ViltForAnswerVocab(backbone_checkpoint=backbone, num_labels=V).to(device)
    state = torch.load(baseline_ckpt, map_location="cpu")
    baseline.load_state_dict(state["model_state_dict"])
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad_(False)

    # KG store/cache/slice cfg
    store = ConceptNetStore(_cfg_get(cfg, ["conceptnet", "db_path"]))
    slice_cache = SliceCache(_cfg_get(cfg, ["kg", "cache_dir"], "data/cache/okvqa/slices"))

    scfg = run_metrics.get("slice_config") or _cfg_get(cfg, ["kg"], {})
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

    # Knowledge encoder must match run settings
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

    # Fusion model + weights
    fusion_ckpt = run_dir / "checkpoints" / "fusion.pt"
    if not fusion_ckpt.exists():
        raise SystemExit(f"Missing {fusion_ckpt}")

    if fusion_mode == "gated":
        fusion = GatedFusion(emb_dim=int(kg_enc.answer_emb.shape[1]), hidden_dim=int(run_metrics.get("hidden_dim") or 128)).to(device)
    else:
        fusion = WeightedAddFusion(alpha_init=0.05, learn_alpha=True).to(device)

    fstate = torch.load(fusion_ckpt, map_location="cpu")["fusion_state_dict"]
    fusion.load_state_dict(fstate, strict=True)
    fusion.eval()

    # Dataset
    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])
    if args.split == "train":
        q = _cfg_get(cfg, ["data", "train_questions_json"])
        a = _cfg_get(cfg, ["data", "train_annotations_json"])
    else:
        q = _cfg_get(cfg, ["data", "val_questions_json"])
        a = _cfg_get(cfg, ["data", "val_annotations_json"])

    ds_base = OKVQADataset(f"{ann_dir}/{q}", f"{ann_dir}/{a}", img_root, load_images=True)
    ds = QAOnlyDataset(ds_base, limit=args.limit)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_fn(processor, b),
    )

    pred_path = out_dir / "predictions.jsonl"
    sel_path = out_dir / "selected_cases.md"
    summary_path = out_dir / "summary.json"

    improved = 0
    worsened = 0
    same = 0
    empty_slices = 0
    items: List[Dict[str, Any]] = []

    with pred_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(dl, desc="dump"):
            b = move_to_device(batch, device)
            base_logits = baseline(b["inputs"]).logits  # [B,V]

            # slices + KG logits
            slices = []
            slice_objs = []
            for qid, iid, qtxt in zip(b["question_ids"], b["image_ids"], b["question_texts"]):
                s, _ = build_slice(store=store, cache=slice_cache, question_id=int(qid), image_id=int(iid), question_text=str(qtxt), cfg=slice_cfg)
                slice_objs.append(s)
                slices.append(s)

            kg = kg_enc.encode_batch(slices)

            if fusion_mode == "gated":
                gate = fusion.mlp(kg.kg_emb)  # [B,1]
                fused_logits = apply_topn_rerank(base_logits, kg.kg_logits, gate, topn)
                scale_used = gate.detach().cpu().view(-1).tolist()
            else:
                a_scale = fusion.alpha()  # scalar >=0
                fused_logits = apply_topn_rerank(base_logits, kg.kg_logits, a_scale, topn)
                scale_used = [float(a_scale.detach().cpu().item())] * base_logits.size(0)

            for i in range(base_logits.size(0)):
                qtxt = b["question_texts"][i]
                gt = batch["answers"][i]
                qid = batch["question_ids"][i]
                iid = batch["image_ids"][i]

                base_top = topk_list(base_logits[i], idx_to_answer, k=args.topk)
                fused_top = topk_list(fused_logits[i], idx_to_answer, k=args.topk)

                base_pred = base_top[0]["answer"]
                fused_pred = fused_top[0]["answer"]

                base_score = vqa_soft_score(base_pred, gt)
                fused_score = vqa_soft_score(fused_pred, gt)

                if fused_score > base_score:
                    improved += 1
                    outcome = "improved"
                elif fused_score < base_score:
                    worsened += 1
                    outcome = "worsened"
                else:
                    same += 1
                    outcome = "same"

                sp = facts_preview(slice_objs[i], max_facts=args.max_facts)
                nonempty = 1 if len(sp) > 0 else 0
                if not nonempty:
                    empty_slices += 1

                conf_base = float(base_top[0]["logit"] - base_top[1]["logit"]) if len(base_top) > 1 else float("nan")
                conf_fused = float(fused_top[0]["logit"] - fused_top[1]["logit"]) if len(fused_top) > 1 else float("nan")

                obj = {
                    "question_id": int(qid),
                    "image_id": int(iid),
                    "question": qtxt,
                    "gt_answers": gt,
                    "baseline": {"pred": base_pred, "soft_score": base_score, "topk": base_top, "margin": conf_base},
                    "fused": {"pred": fused_pred, "soft_score": fused_score, "topk": fused_top, "margin": conf_fused, "scale": float(scale_used[i])},
                    "outcome": outcome,
                    "kg": {
                        "mode": fusion_mode,
                        "topn_rerank": int(topn),
                        "temperature": float(temperature),
                        "slice_config": slice_cfg.__dict__,
                        "nonempty": bool(nonempty),
                        "facts_preview": sp,
                    },
                }
                f.write(json.dumps(obj) + "\n")
                items.append(obj)

    # Select cases for writeup
    def score_delta(x): 
        return x["fused"]["soft_score"] - x["baseline"]["soft_score"]

    improved_cases = sorted([x for x in items if x["outcome"] == "improved"], key=score_delta, reverse=True)[:4]
    worsened_cases = sorted([x for x in items if x["outcome"] == "worsened"], key=score_delta)[:4]
    highconf_wrong = sorted(
        [x for x in items if x["baseline"]["soft_score"] == 0.0],
        key=lambda x: x["baseline"]["margin"],
        reverse=True
    )[:4]

    def case_md(x: Dict[str, Any]) -> str:
        facts = x["kg"]["facts_preview"]
        facts_md = "\n".join([f"- {f['head']} **{f['relation']}** {f['tail']} (score={f['score']})" for f in facts]) if facts else "- (empty slice)"
        return (
            f"### QID {x['question_id']} (img {x['image_id']}) — {x['outcome']}\n\n"
            f"**Q:** {x['question']}\n\n"
            f"**GT answers:** {x['gt_answers']}\n\n"
            f"**Baseline:** {x['baseline']['pred']} (soft={x['baseline']['soft_score']:.3f}, margin={x['baseline']['margin']:.3f})\n\n"
            f"**Fused:** {x['fused']['pred']} (soft={x['fused']['soft_score']:.3f}, margin={x['fused']['margin']:.3f}, scale={x['fused']['scale']:.4f})\n\n"
            f"**KG facts (preview):**\n{facts_md}\n\n"
        )

    md = []
    md.append(f"# Selected Error Analysis Cases ({fusion_mode}, topn={topn}, temp={temperature})\n\n")
    md.append("Generated file. Paste selected cases into the dissertation/poster.\n\n")
    md.append("## Improved cases\n\n")
    for x in improved_cases:
        md.append(case_md(x))
    md.append("## Worsened cases\n\n")
    for x in worsened_cases:
        md.append(case_md(x))
    md.append("## High-confidence wrong baseline predictions\n\n")
    for x in highconf_wrong:
        md.append(case_md(x))
    sel_path.write_text("".join(md), encoding="utf-8")

    summary = {
        "fusion_run_dir": str(run_dir),
        "split": args.split,
        "limit": args.limit,
        "fusion_mode": fusion_mode,
        "topn_rerank": topn,
        "temperature": temperature,
        "counts": {"improved": improved, "worsened": worsened, "same": same, "empty_slices": empty_slices},
        "rates": {
            "improved": improved / max(1, len(items)),
            "worsened": worsened / max(1, len(items)),
            "empty_slices": empty_slices / max(1, len(items)),
        },
        "files": {"predictions_jsonl": str(pred_path), "selected_cases_md": str(sel_path), "summary_json": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote:", pred_path)
    print("Wrote:", sel_path)
    print("Wrote:", summary_path)
    print("Summary:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
