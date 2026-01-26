from __future__ import annotations

import argparse
import time

from run_metadata import init_run, write_metrics
from src.datasets.okvqa import OKVQADataset
from src.eval.normalize import vqa_soft_accuracy
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
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--load_images", type=str, default="true")
    args = ap.parse_args()

    load_images = args.load_images.lower() in ("1", "true", "yes", "y")

    cfg = load_config(args.config)
    out_root = _cfg_get(cfg, ["logging", "out_root"], "experiments/runs")

    ctx = init_run(
        tag=args.tag,
        config_path=args.config,
        out_root=out_root,
        include_pip_freeze=bool(_cfg_get(cfg, ["logging", "include_pip_freeze"], False)),
        extra={"limit": args.limit, "load_images": load_images},
    )

    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])
    split = _cfg_get(cfg, ["data", "split"], "val")

    q_key = "val_questions_json" if split == "val" else "train_questions_json"
    a_key = "val_annotations_json" if split == "val" else "train_annotations_json"
    q_file = _cfg_get(cfg, ["data", q_key])
    a_file = _cfg_get(cfg, ["data", a_key])

    if not all([ann_dir, img_root, q_file, a_file]):
        write_metrics(ctx.run_dir, {"status": "error", "error": "Missing data.* fields for selected split."})
        return 2

    ds = OKVQADataset(
        questions_path=f"{ann_dir}/{q_file}",
        annotations_path=f"{ann_dir}/{a_file}",
        images_root=img_root,
        load_images=load_images,
    )

    model_name = _cfg_get(cfg, ["model", "pretrained_checkpoint"], "dandelin/vilt-b32-finetuned-vqa")

    try:
        from transformers import ViltProcessor, ViltForQuestionAnswering
        import torch
    except Exception as e:
        write_metrics(ctx.run_dir, {"status": "error", "error": f"Missing deps: {type(e).__name__}: {e}"})
        return 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViltProcessor.from_pretrained(model_name)
    model = ViltForQuestionAnswering.from_pretrained(model_name).to(device)
    model.eval()

    n = min(len(ds), args.limit)
    t0 = time.time()

    total = 0
    acc_sum = 0.0

    for i in range(n):
        item = ds[i]
        if not load_images:
            continue

        enc = processor(images=item["image"], text=item["question_text"], return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            pred_idx = int(out.logits.argmax(dim=-1).item())

        pred = model.config.id2label.get(pred_idx, str(pred_idx))
        acc_sum += vqa_soft_accuracy(pred, item["answers"])
        total += 1

    dt = time.time() - t0
    acc = (acc_sum / total) if total > 0 else 0.0

    write_metrics(
        ctx.run_dir,
        {
            "status": "ok",
            "split": split,
            "n_examples": n,
            "n_scored": total,
            "vqa_soft_accuracy_mean": acc,
            "seconds": dt,
            "device": device,
            "checkpoint": model_name,
            "note": "Sanity-check score from pretrained VQA checkpoint, not fine-tuned OK-VQA baseline.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
