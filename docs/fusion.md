# Milestone 5 — Late Fusion (Baseline + KG)

Repo reality check:
- Baseline frozen: `BASELINE_FREEZE_20260312_1456` (val VQA soft acc ~0.1639)
- KG slicing: working + caching validated + tests pass

This milestone adds:
- Knowledge encoder producing `kg_logits` over the 10k answer vocab
- Two fusion strategies (weighted, gated)
- Training/eval script that reports baseline vs fused + delta

## Run a sanity check
```bash
python -m src.train_fusion --config configs/fusion_train.yaml --tag m5_fusion_sanity
cat experiments/runs/*m5_fusion_sanity*/metrics.json
```

## Scale up (full train/val)
Edit `configs/fusion_train.yaml`:
- `limit_train: 9009`
- `limit_val: 5046`

Then rerun with:
- weighted fusion tag
- gated fusion tag
