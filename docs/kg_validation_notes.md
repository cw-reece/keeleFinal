# Knowledge Module Validation Notes (Milestone 4)

This file documents validation checks for the ConceptNet slicing module.

## Guarantees validated

1) Slice bounds
- Slices contain at most `top_k` facts.
- Verified via unit tests and p95 statistics hitting the cap.

2) Determinism + caching
- Cache key includes a config hash and (question_id, image_id).
- Repeated runs with identical config produce cache hits and identical slices.

3) Relation filtering
- Strict relation set enforced (e.g., IsA, UsedFor, CapableOf, AtLocation, etc.).
- All returned facts conform to the selected relation set.

4) Auditable outputs
- Each run writes:
  - `slice_stats.json` (aggregate stats)
  - `slice_samples/` (human-readable samples)

## Scoring / ranking behavior (current)

Scorer version: `v1` (heuristic)
- Base term: ConceptNet weight
- Additive lexical overlap term between question tokens and fact tokens
- Small boost for shared token count
- Optional hop-2 dampening (when hop_depth=2)

Known limitations:
- Entity extraction is heuristic (no POS tagging / lemmatization beyond normalization).
- Scorer is lexical; it does not use embeddings or learned reranking.
- hop_depth > 1 may add noise and should be evaluated via ablation.

## How to run tests

Install pytest if needed:
```bash
pip install pytest
```

Run:
```bash
pytest -q
```
