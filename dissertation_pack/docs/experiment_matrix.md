# Milestone 6 — Automated Experiment Matrix

## Run the matrix (small limits)
```bash
python -m scripts.run_matrix --matrix configs/experiment_matrix.yaml
```

Generated configs:
- `configs/generated/matrix/`

Runs:
- `experiments/runs/`

## Summarize results
```bash
python -m scripts.summarize_runs --out_dir reports --filter m5_ --filter m6_
```

Outputs:
- `reports/run_summary.csv`
- `reports/run_summary.md`

## Plot quick visuals
```bash
python -m scripts.plot_results --csv reports/run_summary.csv --out_dir reports/plots
```

## Scale up later
Edit `configs/experiment_matrix.yaml`:
- `training.limit_train: 9009`
- `training.limit_val: 5046`
Optionally set `training.epochs: 2` or `3`.
