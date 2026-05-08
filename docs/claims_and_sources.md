# Claims and Sources (traceability map)

**Date:** 2026-03-19

Rule: every headline claim in the dissertation must point to a concrete file in the repo.

## Baseline

Claim:
- Baseline validation VQA soft accuracy = **0.16389219183511633**

Source:
- `experiments/runs/BASELINE_FREEZE_20260312_1456/metrics.json`
- `experiments/runs/BASELINE_FREEZE_20260312_1456/checkpoints/model.pt`

Supporting (coverage):
- Vocab size = 10,000
- Train coverage = 100%
- Val coverage ≈ 96.5%

Source:
- output of `python -m scripts.vocab_coverage --config configs/baseline.yaml --vocab data/processed/okvqa/answer_vocab.json`

## KG slicing (ConceptNet)

Claims (example, update to match the specific run you cite):
- Slice nonempty rate ~ 0.80 (on subset)
- Mean build time ~ 1–2 ms
- Facts per slice bounded by top_k

Sources:
- `experiments/runs/*m3_*/slice_stats.json`
- `configs/kg_slice.yaml`

## Late fusion (Milestone 5)

Claim:
- Naive global weighted fusion degrades accuracy substantially (Δ ≈ -0.0534).

Source:
- `experiments/runs/20260313_140303_m5_fusion_weighted_fullval/metrics.json`

Claim:
- Gated fusion often learns to ignore KG (Δ = 0).

Source:
- `experiments/runs/20260313_141000_m5_gated_fullval/metrics.json`
- `experiments/runs/20260313_145705_m5_topn20_gated_fullval/metrics.json`

Claim:
- Stabilized top-N weighted fusion is safe and near-baseline (Δ ≈ -0.000132).

Source:
- `experiments/runs/20260313_145031_m5_topn20_weighted_fullval/metrics.json`

## Experiment matrix (Milestone 6)

Claim:
- Automated matrix produced summary tables + plots.

Sources:
- `reports/run_summary.csv`
- `reports/run_summary.md`
- `reports/plots/delta_hist.png`
- `reports/plots/delta_vs_topk.png`
- `configs/experiment_matrix.yaml`

## Error analysis

Claim:
- Gated top-N run: 300/300 examples unchanged; empty slice rate ≈ 21.7% (65/300).

Source:
- `reports/error_analysis/m5_topn20_gated_val300/summary.json`

Claim:
- Failure modes include empty slices, generic entity extraction (“type”), wrong neighborhood, visual grounding gap, answer mismatch.

Sources:
- `reports/error_analysis/m5_topn20_weighted_val300/selected_cases.md`
- `reports/error_analysis/m5_topn20_gated_val300/selected_cases.md`
