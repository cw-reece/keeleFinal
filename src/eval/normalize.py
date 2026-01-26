from __future__ import annotations

import re
import string

_ARTICLES = {"a", "an", "the"}
_PUNCT = set(string.punctuation)

_NUM_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_answer(ans: str, map_numbers: bool = True) -> str:
    """
    Conservative normalization for comparing answers.
    Not guaranteed identical to official OK-VQA eval, but consistent for early checks.
    """
    if ans is None:
        return ""
    s = ans.strip().lower()

    if map_numbers:
        toks = s.split()
        toks = [_NUM_WORDS.get(t, t) for t in toks]
        s = " ".join(toks)

    s = "".join(ch for ch in s if ch not in _PUNCT)
    toks = [t for t in s.split() if t not in _ARTICLES]
    s = re.sub(r"\s+", " ", " ".join(toks)).strip()
    return s


def vqa_soft_accuracy(pred: str, gt_answers: list[str]) -> float:
    """
    Simple VQA-style soft accuracy approximation:
    score = min(count(pred in gt)/3, 1).
    """
    p = normalize_answer(pred)
    gts = [normalize_answer(a) for a in gt_answers]
    matches = sum(1 for a in gts if a == p)
    return min(matches / 3.0, 1.0)
