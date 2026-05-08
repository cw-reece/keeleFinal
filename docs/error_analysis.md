# Error analysis workflow

Goal: produce qualitative examples for the report/poster explaining why KG helps/hurts/is ignored.

## Recommended runs to analyze
- Weighted Top-N: `experiments/runs/20260313_145031_m5_topn20_weighted_fullval`
- Gated Top-N: `experiments/runs/20260313_145705_m5_topn20_gated_fullval`
- (Optional contrast) Global weighted: `experiments/runs/20260313_140303_m5_fusion_weighted_fullval`

## Run (val subset)
```bash
python -m scripts.error_analysis_dump \
  --config configs/fusion_train_v3_topn.yaml \
  --fusion_run_dir experiments/runs/20260313_145031_m5_topn20_weighted_fullval \
  --split val --limit 300 \
  --out_dir reports/error_analysis/m5_topn20_weighted_val300
```

Then open:
- `reports/error_analysis/.../selected_cases.md`

## What to look for
- Entity extraction failures (missed key concept, wrong n-gram).
- Slice facts irrelevant to the question (noisy neighbors).
- Facts correct but not aligned with answer vocab strings.
- Baseline confidently wrong + KG facts plausible but fusion ignored (alignment/scoring bottleneck).
