# What to include in the dissertation (copy/paste checklist)

## Core (must include)
- Problem statement + research question + hypothesis
- Dataset: OK-VQA (train/val splits used), images source
- Primary metric: VQA-style soft accuracy
- Baseline model:
  - ViLT backbone + initialization choice
  - answer vocabulary creation (size 10k) + coverage numbers
  - training settings (batch size, epochs, loss, pos_weight, seed)
  - best run ID + val result table
- Knowledge pipeline:
  - ConceptNet ingestion into SQLite (what, why, deterministic indexing)
  - entity extraction approach (n-grams, max_entities, stopwords)
  - slice construction: hop depth, relation filters, neighbor limits, top_k
  - caching key and determinism rationale
  - slice stats (nonempty rate, facts per slice, build time)
- Late fusion:
  - naive global fusion result (show it can hurt)
  - stabilized fusion (alpha constrained, top-N rerank)
  - gated fusion behavior (“safety gate”)
- Automated evaluation harness (matrix):
  - how the grid is defined
  - how results are summarized into tables/plots
- Error analysis:
  - at least 6–12 cases with KG facts shown
  - categorize failure modes (empty slice, generic entity, wrong neighborhood, visual grounding gap, answer mismatch)
- Limitations + threats to validity
- Reproducibility details:
  - run folder structure, config snapshots, seeds, checkpoints

## Artifacts (figures/tables)
- Baseline results table
- KG slicing stats table
- Fusion comparison table (baseline vs fused)
- Matrix summary table + 1–2 plots (delta histogram, delta vs top_k)
- Architecture diagram (even simple)
- 1–2 pages error analysis examples

## Appendix (optional but helpful)
- exact config files used
- links/QR to repo + reproduction commands
