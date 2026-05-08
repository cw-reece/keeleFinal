# Dissertation bundle — what to do next (step-by-step)

You’re not behind; you’re at the point where the work shifts from *building* to *packaging + writing*.

## Step 0 — Don’t change the baseline
Your baseline is frozen: `BASELINE_FREEZE_20260312_1456` (val VQA soft acc ~0.163892).
Do not retrain baseline unless you decide to explicitly create “Baseline v2”.

## Step 1 — Generate the remaining error analysis artifact(s)
You already generated weighted top-N error analysis (subset).

Do the gated one (subset 300) so you can compare “KG ignored” behavior:

```bash
python -m scripts.error_analysis_dump \
  --config configs/fusion_train_v3_topn.yaml \
  --fusion_run_dir experiments/runs/20260313_145705_m5_topn20_gated_fullval \
  --split val --limit 300 \
  --out_dir reports/error_analysis/m5_topn20_gated_val300
```

Optional (contrast / “why naive fusion is dangerous”):
```bash
python -m scripts.error_analysis_dump \
  --config configs/fusion_train_v3_topn.yaml \
  --fusion_run_dir experiments/runs/20260313_140303_m5_fusion_weighted_fullval \
  --split val --limit 200 \
  --out_dir reports/error_analysis/m5_global_weighted_val200
```

## Step 2 — Build your “dissertation pack” ZIP from the repo (one command)
This bundle includes a packaging script. Run it from repo root:

```bash
python tools/pack_repo_artifacts.py --repo_root . --out_dir dissertation_pack
```

It will:
- collect key docs, configs, run metrics, plots, and selected error-analysis cases
- write a manifest noting what was found/missing
- create `dissertation_pack_YYYYMMDD.zip` you can upload/submit or keep for writing.

## Step 3 — Paste-ready tables + plots
Use:
- `reports/run_summary.md` as your master experiment table
- `reports/plots/delta_hist.png` and `reports/plots/delta_vs_topk.png` in your Results section
- `docs/fusion_results.md` (or the fusion table you created) as the “Milestone 5/6” summary

## Step 4 — Write the dissertation in this order (fastest path)
1) **Method** (what you built)
2) **Experimental setup** (data splits, metric, training settings, reproducibility)
3) **Results** (baseline, slicing stats, fusion outcomes, matrix results)
4) **Error analysis** (6–12 cases)
5) **Limitations & threats to validity**
6) **Conclusion & future work**
Then do Intro/Abstract last.

## Step 5 — What to do if your results are “no gain”
Write it as engineering science:
- show naive fusion hurts (global fusion)
- show constrained fusion is safe and stable
- explain why KG didn’t help (entity extraction brittleness, slice relevance, answer-space mismatch)
- propose next improvements (visual concepts, learned fact scoring, cross-encoder)

That’s a legitimate MSc outcome and it reads like competent research.
