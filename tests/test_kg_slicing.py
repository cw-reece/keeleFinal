# tests/test_kg_slicing.py
import pytest
from pathlib import Path

from src.utils.config import load_config
from src.datasets.okvqa import OKVQADataset
from src.kg.conceptnet_store import ConceptNetStore
from src.kg.cache import SliceCache
from src.kg.slice_builder import SliceConfig, build_slice, config_hash
from src.kg.relation_filter import STRICT_RELATIONS


CFG_PATH = Path("configs/kg_slice.yaml")


def _cfg_get(cfg: dict, path: list[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _require_paths():
    if not CFG_PATH.exists():
        pytest.skip("configs/kg_slice.yaml not found")
    cfg = load_config(str(CFG_PATH))

    db_path = Path(_cfg_get(cfg, ["conceptnet", "db_path"], ""))
    if not db_path.exists():
        pytest.skip(f"ConceptNet DB missing: {db_path}")

    ann_dir = Path(_cfg_get(cfg, ["data", "annotations_dir"], ""))
    q_file = _cfg_get(cfg, ["data", "val_questions_json"], "")
    a_file = _cfg_get(cfg, ["data", "val_annotations_json"], "")

    if not (ann_dir / q_file).exists() or not (ann_dir / a_file).exists():
        pytest.skip("OK-VQA JSONs missing (val)")

    return cfg


def _one_val_example(cfg: dict):
    ann_dir = _cfg_get(cfg, ["data", "annotations_dir"])
    img_root = _cfg_get(cfg, ["data", "coco_images_root"])
    q_file = _cfg_get(cfg, ["data", "val_questions_json"])
    a_file = _cfg_get(cfg, ["data", "val_annotations_json"])
    ds = OKVQADataset(f"{ann_dir}/{q_file}", f"{ann_dir}/{a_file}", img_root, load_images=False)
    item = ds[0]
    return int(item["question_id"]), int(item["image_id"]), str(item["question_text"])


def test_slice_topk_bound():
    cfg = _require_paths()
    store = ConceptNetStore(_cfg_get(cfg, ["conceptnet", "db_path"]))
    cache = SliceCache("data/cache/okvqa/slices_test")

    qid, iid, qtext = _one_val_example(cfg)
    scfg = SliceConfig(top_k=5, hop_depth=1, relation_set="strict")

    s, _ = build_slice(store=store, cache=cache, question_id=qid, image_id=iid, question_text=qtext, cfg=scfg)
    assert len(s["facts"]) <= 5
    assert int(s["stats"]["n_facts"]) <= 5


def test_relation_filter_enforced_strict():
    cfg = _require_paths()
    store = ConceptNetStore(_cfg_get(cfg, ["conceptnet", "db_path"]))
    cache = SliceCache("data/cache/okvqa/slices_test")

    qid, iid, qtext = _one_val_example(cfg)
    scfg = SliceConfig(top_k=10, hop_depth=1, relation_set="strict")

    s, _ = build_slice(store=store, cache=cache, question_id=qid, image_id=iid, question_text=qtext, cfg=scfg)
    for f in s["facts"]:
        assert f["relation"] in STRICT_RELATIONS


def test_cache_hit_and_determinism():
    cfg = _require_paths()
    store = ConceptNetStore(_cfg_get(cfg, ["conceptnet", "db_path"]))
    cache = SliceCache("data/cache/okvqa/slices_test")

    qid, iid, qtext = _one_val_example(cfg)
    scfg = SliceConfig(top_k=10, hop_depth=1, relation_set="strict")

    s1, _ = build_slice(store=store, cache=cache, question_id=qid, image_id=iid, question_text=qtext, cfg=scfg)
    s2, hit2 = build_slice(store=store, cache=cache, question_id=qid, image_id=iid, question_text=qtext, cfg=scfg)

    assert hit2 is True
    assert s1["stats"]["config_hash"] == s2["stats"]["config_hash"]
    assert s1["facts"] == s2["facts"]
    assert s1["entities"] == s2["entities"]


def test_config_hash_changes_with_knobs():
    a = config_hash(SliceConfig(top_k=10, hop_depth=1, relation_set="strict"))
    b = config_hash(SliceConfig(top_k=20, hop_depth=1, relation_set="strict"))
    c = config_hash(SliceConfig(top_k=10, hop_depth=1, relation_set="broad"))
    assert a != b
    assert a != c
