# KG Slicing Results Summary (Milestone 3)

This document records the observed behavior of the task-specific ConceptNet slicing pipeline on OK-VQA **val**.

## What was validated

- **Indexing / queryability:** ConceptNet assertions ingested into SQLite and neighbor queries return quickly.
- **Bounded slicing:** slice size is bounded by `top_k` (median and p95 hit exactly the cap when nonempty).
- **Caching:** rerunning the same slice config produces **cache_hit_rate = 1.0**.
- **Auditability:** `slice_samples/` markdown files exist per run and can be inspected manually.

## Results (val)

| Run ID | Split | Relations | top_k | hop_depth | N | Nonempty rate | Facts per slice (mean/median/p95) | Build ms (mean/median/p95) | Cache hit rate |
| --- | --- | --- | ---:| ---:| ---:| ---:| --- | --- | ---:|
| 20260304_144521_m3_slices_val_200 | val | strict | 10 | 1 | 200 | 0.800 | 7.455 / 10 / 10 | 1.693 / 1.499 / 3.450 | 0.0 |
| 20260304_144739_m3_slices_val_200_rerun | val | strict | 10 | 1 | 200 | 0.800 | 7.455 / 10 / 10 | 1.693 / 1.499 / 3.450 | 1.0 |
| 20260304_144935_m3_strict_top5 | val | strict | 5 | 1 | 500 | 0.794 | 3.794 / 5 / 5 | 1.698 / 1.487 / 3.447 | 0.0 |
| 20260304_145124_m3_strict_top20 | val | strict | 20 | 1 | 500 | 0.794 | 14.616 / 20 / 20 | 1.698 / 1.494 / 3.474 | 0.0 |
| 20260304_145347_m3_broad_top10 | val | broad | 20* | 1 | 500 | 0.796 | 14.622 / 20 / 20 | 1.787 / 1.597 / 3.675 | 0.0 |

\* Note: the run tag says `broad_top10` but the saved config shows `top_k=20`. Treat config as authoritative.

## Interpretation

- Increasing `top_k` increases mean facts per slice as expected (bounded at the cap).
- Relation set (strict vs broad) slightly increases build time and can increase slice size when coupled with larger `top_k`.
- Nonempty rate is stable (~0.79–0.80) across these settings on the tested sample sizes.

## Next steps (Milestone 4)

- Add tests covering:
  - top_k bounds
  - relation filtering enforcement
  - determinism via cache
  - config-hash changes on knob changes
- Add validation notes describing retrieval/scoring behavior and known limitations.
