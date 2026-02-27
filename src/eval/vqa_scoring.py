# src/eval/vqa_scoring.py
from __future__ import annotations

from src.eval.normalize import vqa_soft_accuracy


def mean_vqa_soft_accuracy(pred_answers: list[str], gt_answers: list[list[str]]) -> float:
    if not pred_answers:
        return 0.0
    total = 0.0
    for p, gts in zip(pred_answers, gt_answers):
        total += vqa_soft_accuracy(p, gts)
    return total / float(len(pred_answers))
