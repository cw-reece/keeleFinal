#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports

LOG="reports/qc_log.md"
echo "# QC Log" > "$LOG"
echo "" >> "$LOG"
echo "Generated: $(date)" >> "$LOG"
echo "" >> "$LOG"

run() {
  echo "" | tee -a "$LOG"
  echo "## $*" | tee -a "$LOG"
  echo "" | tee -a "$LOG"
  ( "$@" ) 2>&1 | tee -a "$LOG"
}

run pytest -q
run python -m scripts.data_check_okvqa --config configs/baseline.yaml --max_image_checks 200
run python -m scripts.vocab_coverage --config configs/baseline.yaml --vocab data/processed/okvqa/answer_vocab.json
run python -m scripts.eval_baseline_ckpt --config configs/baseline_train_v4_suggested.yaml --checkpoint experiments/runs/BASELINE_FREEZE_20260312_1456/checkpoints/model.pt --split val
run python -m scripts.eval_fusion_run --config configs/fusion_train_v3_topn.yaml --fusion_run_dir experiments/runs/20260313_145031_m5_topn20_weighted_fullval --split val
run python -m scripts.eval_fusion_run --config configs/fusion_train_v3_topn.yaml --fusion_run_dir experiments/runs/20260313_145705_m5_topn20_gated_fullval --split val
run python -m scripts.summarize_runs --out_dir reports --filter m5_ --filter m6_
run python -m scripts.plot_results --csv reports/run_summary.csv --out_dir reports/plots

echo "" | tee -a "$LOG"
echo "QC complete. Log written to $LOG" | tee -a "$LOG"
