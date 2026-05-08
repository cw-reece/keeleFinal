# Results section skeleton (paste-ready headings)

## Baseline performance
- Describe baseline configuration (ViLT backbone, VQA initialization, answer vocab 10k, loss/pos_weight)
- Report: best baseline run ID and validation VQA soft accuracy
- Note coverage: train 100%, val 96.5% for 10k vocab

## Knowledge graph slicing (ConceptNet)
- Report slicing configuration(s) (hop, top_k, relation_set, neighbor_limit)
- Slice coverage: nonempty rate
- Slice size stats: mean/median/p95 facts per slice
- Runtime: build_ms mean/median/p95
- Caching: cache hit rate on rerun

## Late fusion results
- Show naive fusion over full answer space can degrade performance substantially
- Show stabilized top-N constrained fusion is safe (near-baseline delta)
- Show gated fusion typically suppresses KG (delta ~0)

## Experiment matrix (automated)
- Describe grid (relation_set × top_k × fusion mode)
- Summarize key patterns:
  - gated ≈ baseline
  - weighted tends slightly negative unless strongly constrained
  - no consistent positive deltas observed under current KG encoding

## Error analysis (qualitative)
- Provide 6–12 example cases
- Group into failure modes:
  - empty slice
  - generic entity captured (“type”)
  - wrong concept neighborhood
  - needs visual grounding
  - answer-space mismatch
- Explain why these predict the observed deltas
