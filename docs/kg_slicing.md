# Milestone 3 — ConceptNet Ingestion + Task-Specific Slicing

This repo supports Milestone 3 by:
- ingesting ConceptNet assertions into a local SQLite DB for fast neighbor lookup
- extracting entities from OK-VQA questions (cheap, deterministic heuristic)
- building a bounded “slice” per (image, question) by retrieving and ranking facts
- caching slices to disk (reproducibility + speed)
- emitting an auditable sample set + slice statistics for the report

## 1) Download ConceptNet assertions
Download **ConceptNet 5.7.0 assertions** and place it at:

`data/raw/conceptnet/conceptnet-assertions-5.7.0.csv.gz`

## 2) Build the SQLite index
```bash
python -m scripts.build_conceptnet_db --config configs/kg_slice.yaml
```

This creates:
`data/processed/conceptnet/conceptnet_en.sqlite`

## 3) Build slices (cached) and produce audit samples
```bash
python -m scripts.build_slices --config configs/kg_slice.yaml --split val --limit 200 --tag m3_slices_val_200
```

Outputs:
- cached slices under `data/cache/okvqa/slices/<config_hash>/`
- run artifacts under `experiments/runs/<run_id>/` including:
  - `slice_stats.json`
  - `slice_samples/` (markdown files)

## 4) What to report
From `slice_stats.json`, report:
- nonempty slice rate
- facts per slice (mean/median/p95)
- avg slice build time
- cache hit rate
