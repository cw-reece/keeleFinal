from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass(frozen=True)
class OKVQARecord:
    question_id: int
    image_id: int
    question: str
    answers: List[str]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index_annotations(ann_json: Dict[str, Any]) -> Dict[int, List[str]]:
    """Returns: {question_id: [answers...]}"""
    idx: Dict[int, List[str]] = {}
    annotations = ann_json.get("annotations", [])
    for a in annotations:
        qid = int(a["question_id"])
        answers: List[str] = []

        if "answers" in a and isinstance(a["answers"], list):
            # VQA format: [{"answer": "..."} ...]
            if a["answers"] and isinstance(a["answers"][0], dict) and "answer" in a["answers"][0]:
                answers = [str(x.get("answer", "")) for x in a["answers"]]
            else:
                answers = [str(x) for x in a["answers"]]
        elif "multiple_choice_answer" in a:
            answers = [str(a["multiple_choice_answer"])]

        idx[qid] = answers
    return idx


def _resolve_image_path(images_root: Path, image_id: int) -> Optional[Path]:
    """
    OK-VQA uses COCO 2014 images. Try common naming patterns.
    Supports:
      - images_root/train2014 + images_root/val2014
      - or images directly in images_root
    """
    fname_train = f"COCO_train2014_{image_id:012d}.jpg"
    fname_val = f"COCO_val2014_{image_id:012d}.jpg"

    candidates: list[Path] = []
    if (images_root / "train2014").exists():
        candidates.append(images_root / "train2014" / fname_train)
    if (images_root / "val2014").exists():
        candidates.append(images_root / "val2014" / fname_val)

    candidates.append(images_root / fname_train)
    candidates.append(images_root / fname_val)

    for p in candidates:
        if p.exists():
            return p
    return None


class OKVQADataset:
    """
    Minimal OK-VQA loader for early Milestone 2.

    Returns “contract fields”:
      - image (PIL.Image) OR image_path (str/None)
      - question_text (str)
      - answers (list[str])
      - question_id (int)
      - image_id (int)
    """

    def __init__(
        self,
        questions_path: str | Path,
        annotations_path: str | Path,
        images_root: str | Path,
        *,
        load_images: bool = False,
    ) -> None:
        self.questions_path = Path(questions_path)
        self.annotations_path = Path(annotations_path)
        self.images_root = Path(images_root)
        self.load_images = load_images

        q_json = _read_json(self.questions_path)
        a_json = _read_json(self.annotations_path)

        self._ann_by_qid = _index_annotations(a_json)

        questions = q_json.get("questions", [])
        self.records: List[OKVQARecord] = []
        for q in questions:
            qid = int(q["question_id"])
            img_id = int(q["image_id"])
            text = str(q.get("question", ""))
            answers = self._ann_by_qid.get(qid, [])
            self.records.append(
                OKVQARecord(question_id=qid, image_id=img_id, question=text, answers=answers)
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        img_path = _resolve_image_path(self.images_root, r.image_id)

        item: Dict[str, Any] = {
            "question_id": r.question_id,
            "image_id": r.image_id,
            "question_text": r.question,
            "answers": r.answers,
        }

        if self.load_images:
            if img_path is None:
                raise FileNotFoundError(f"Image not found for image_id={r.image_id} under {self.images_root}")
            item["image"] = Image.open(img_path).convert("RGB")
        else:
            item["image_path"] = str(img_path) if img_path is not None else None

        return item


def list_json_files(dir_path: str | Path) -> List[str]:
    p = Path(dir_path)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob("*.json")])
