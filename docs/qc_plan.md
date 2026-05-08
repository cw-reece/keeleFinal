# QC Plan (what to run before submission)

Goal: high confidence that (1) numbers are traceable, (2) key results reproduce, (3) no silent dataset/config issues.

## Quick checklist (run in this order)

1) Unit tests
- `pytest -q`

2) Dataset integrity
- `python -m scripts.data_check_okvqa --config configs/baseline.yaml --max_image_checks 200`

3) Vocab coverage
- `python -m scripts.vocab_coverage --config configs/baseline.yaml --vocab data/processed/okvqa/answer_vocab.json`

4) Baseline eval-from-checkpoint (full val)
- `python -m scripts.eval_baseline_ckpt --config configs/baseline_train_v4_suggested.yaml --checkpoint experiments/runs/BASELINE_FREEZE_20260312_1456/checkpoints/model.pt --split val`

5) Fusion eval-from-checkpoint (full val)
- `python -m scripts.eval_fusion_run --config configs/fusion_train_v3_topn.yaml --fusion_run_dir experiments/runs/20260313_145031_m5_topn20_weighted_fullval --split val`
- `python -m scripts.eval_fusion_run --config configs/fusion_train_v3_topn.yaml --fusion_run_dir experiments/runs/20260313_145705_m5_topn20_gated_fullval --split val`

6) Matrix summary reproducibility
- `python -m scripts.summarize_runs --out_dir reports --filter m5_ --filter m6_`
- `python -m scripts.plot_results --csv reports/run_summary.csv --out_dir reports/plots`

## Output logging
Use the provided script:
- `bash tools/qc_run_all.sh`

It writes `reports/qc_log.md`.
