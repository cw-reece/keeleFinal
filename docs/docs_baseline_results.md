# Baseline Results (OK-VQA)

**Project:** Knowledge-Augmented VQA via Task-Specific KG Slicing + Late Fusion  
**Student:** Christopher Ward Reece  
**Last updated:** 2026-03-04

This file summarizes baseline-related runs that are logged under `experiments/runs/<run_id>/`.

# Baseline Results (OK-VQA)

**Project:** Knowledge-Augmented VQA via Task-Specific KG Slicing + Late Fusion  
**Student:** Christopher Ward Reece  
**Last updated:** 2026-05-16

This file summarizes baseline-related runs and identifies the frozen baseline used for final matched comparison.

## Final frozen baseline used for dissertation reporting

The final frozen baseline used in the main fusion comparisons is:

- **Run ID:** `BASELINE_FREEZE_20260312_1456`
- **Checkpoint path:** `experiments/runs/20260304_123946_baseline_v4_ep5/checkpoints/model.pt`
- **Setup:** ViLT encoder initialized from `dandelin/vilt-b32-finetuned-vqa`, answer vocabulary size 10,000, BCE loss with `pos_weight=20`, mixed precision enabled.
- **Split:** OK-VQA validation, 5046 examples.
- **Metric:** VQA-style soft accuracy.
- **Result:** **0.163892**

This is the authoritative baseline for final dissertation comparisons.

## Earlier baseline development runs

The earlier `20260304_092423_baseline_v4_pw20_ep3_fullval` run achieved `0.056084`. It is retained as development history, but it is not the final frozen baseline used for the final fusion comparisons.
## Run summary table

| Run ID | Tag | Epochs | Batch | Train N | Val N | Vocab | Val VQA soft acc | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260126_1508_m2_train_stub | m2_train_stub |  |  | 32 |  |  |  | Training stub only (proof of run logging). |
| 20260126_1515_m2_eval_pretrained | m2_eval_pretrained |  |  |  |  |  |  | Pretrained ViLT VQA sanity-check (not a trained OK-VQA baseline). |
| 20260227_1230_baseline_v1_ep1 | baseline_v1_ep1 | 1 | 4 | 256 | 256 | 3000 |  | First end-to-end baseline trainer smoke run (tiny limits). |
| 20260302_093620_baseline_v2_vocab10k_ep3_fullval | baseline_v2_vocab10k_ep3_fullval | 3 | 4 | 9009 | 5046 | 10000 | 0.001850 | Vocab=10k, full train/val, MLM-init (collapsed; near-zero accuracy). |
| 20260304_090407_baseline_v4_pw20_quick | baseline_v4_pw20_quick | 2 | 4 | 2048 | 2048 | 10000 | 0.004232 | VQA-init + pos_weight=20 quickcheck (train/val=2048) shows learning signal. |
| 20260304_092423_baseline_v4_pw20_ep3_fullval | baseline_v4_pw20_ep3_fullval | 2 | 3 | 9009 | 5046 | 10000 | 0.056084 | Earlier baseline development run; not used as the final frozen baseline.

## Reproduce the current baseline run

1) Ensure dataset integrity passes:
```bash
python -m scripts.data_check_okvqa --config configs/baseline.yaml --max_image_checks 200
```

2) Ensure vocab is 10k and coverage is high:
```bash
python -m scripts.build_answer_vocab --config configs/baseline.yaml --top_n 10000
python -m scripts.vocab_coverage --config configs/baseline.yaml --vocab data/processed/okvqa/answer_vocab.json
```

3) Train baseline:
```bash
python -m src.train_baseline --config configs/baseline_train_v4_suggested.yaml --tag baseline_v4_pw20_epX_fullval
```

## Next recommended run (optional)

If you want to tighten the baseline number:
- run **epochs=3** or **epochs=5** with the same settings (VQA-init + pos_weight=20),
- log both runs and update this table with the best run_id.

