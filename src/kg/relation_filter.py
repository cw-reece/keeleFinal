# src/kg/relation_filter.py
from __future__ import annotations

STRICT_RELATIONS = {
    "IsA",
    "UsedFor",
    "CapableOf",
    "AtLocation",
    "PartOf",
    "HasA",
    "HasProperty",
    "MadeOf",
    "Causes",
    "CreatedBy",
    "DefinedAs",
    "LocatedNear",
    "SimilarTo",
    "Synonym",
    "Antonym",
    "RelatedTo",
}

BROAD_RELATIONS = STRICT_RELATIONS | {
    "HasSubevent",
    "HasFirstSubevent",
    "HasLastSubevent",
    "MotivatedByGoal",
    "CausesDesire",
    "Desires",
    "NotDesires",
    "DistinctFrom",
    "EtymologicallyRelatedTo",
    "EtymologicallyDerivedFrom",
}


def relation_set(name: str) -> set[str]:
    n = (name or "").strip().lower()
    if n in ("strict", "default", ""):
        return set(STRICT_RELATIONS)
    if n in ("broad", "all"):
        return set(BROAD_RELATIONS)
    # allow comma-separated custom
    if "," in n:
        return set(x.strip() for x in name.split(",") if x.strip())
    return set(STRICT_RELATIONS)
