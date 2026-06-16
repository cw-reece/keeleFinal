# Appendices

## Appendix A: Project Artefact Index

This appendix identifies the main repository artefacts used to support the dissertation. The purpose is to make the submission traceable: each major claim in the report should be linked to a code component, configuration file, run output, or summary report.

| Artefact | Purpose |
|---|---|
| `README.md` | Examiner-facing project overview, setup notes, reproduction commands, and headline result. |
| `docs/docs_baseline_results.md` | Baseline development history and final frozen baseline identification. |
| `docs/docs_architecture.md` | Architecture notes for the OK-VQA, ConceptNet slicing, and late-fusion pipeline. |
| `docs/docs_risks.md` | Risk register and project management evidence. |
| `reports/final_results/final_results_summary.md` | Final quantitative result summary used in the dissertation. |
| `reports/final_results/random_slice_control.md` | Random-slice control result and interpretation. |
| `reports/final_results/run_summary.md` | Summary of saved fusion runs. |
| `reports/final_results/run_summary.csv` | Machine-readable run summary table. |
| `reports/qc_log.md` | Quality-control log for baseline and fusion checks. |
| `dissertation_pack/experiments/runs/` | Packaged run metrics used for dissertation reporting. |
| `configs/` | Baseline, fusion, ablation, and control configurations. |
| `src/` | Main implementation package. |
| `scripts/` | Data checks, slice building, evaluation, summarisation, and error-analysis scripts. |

## Appendix B: Final Run ID Table

The final dissertation reports the following headline runs.

| Run ID | Description | Validation examples | Baseline accuracy | Fused accuracy | Delta |
|---|---|---:|---:|---:|---:|
| `BASELINE_FREEZE_20260312_1456` | Frozen ViLT baseline used for matched comparison | 5046 | 0.163892 | n/a | n/a |
| `20260313_140303_m5_fusion_weighted_fullval` | Naive full-space weighted KG fusion | 5046 | 0.163892 | 0.110517 | -0.053376 |
| `20260313_141000_m5_gated_fullval` | Naive gated KG fusion | 5046 | 0.163892 | 0.163892 | 0.000000 |
| `20260313_143730_m5_topn_weighted_fullval` | Top-50 constrained weighted KG fusion | 5046 | 0.163892 | 0.163760 | -0.000132 |
| `20260313_145031_m5_topn20_weighted_fullval` | Top-20 constrained weighted KG fusion | 5046 | 0.163892 | 0.163760 | -0.000132 |
| `20260313_145705_m5_topn20_gated_fullval` | Top-20 constrained gated KG fusion | 5046 | 0.163892 | 0.163892 | 0.000000 |
| `m6_random_slice_weighted_topn20_fullval` | Random-slice weighted control | 5046 | 0.163892 | 0.163760 | -0.000132 |

The final result does not show an accuracy improvement from bounded ConceptNet late fusion. Instead, the results show that naive weighted fusion can harm performance, top-N constrained fusion reduces the harm, gated fusion preserves the frozen baseline, and the random-slice control does not create a spurious gain.

## Appendix C: Ablation Matrix

The ablation matrix used a 512-example validation subset. These runs are diagnostic and should not be treated as headline performance evidence.

| Fusion mode | Relation set | Top-K | Validation examples | Baseline accuracy | Fused accuracy | Delta |
|---|---|---:|---:|---:|---:|---:|
| weighted | strict | 5 | 512 | 0.135417 | 0.134115 | -0.001302 |
| weighted | strict | 10 | 512 | 0.135417 | 0.134766 | -0.000651 |
| weighted | strict | 20 | 512 | 0.135417 | 0.134115 | -0.001302 |
| gated | strict | 5 | 512 | 0.135417 | 0.134766 | -0.000651 |
| gated | strict | 10 | 512 | 0.135417 | 0.134766 | -0.000651 |
| gated | strict | 20 | 512 | 0.135417 | 0.135417 | 0.000000 |
| weighted | broad | 5 | 512 | 0.135417 | 0.134115 | -0.001302 |
| weighted | broad | 10 | 512 | 0.135417 | 0.134766 | -0.000651 |
| weighted | broad | 20 | 512 | 0.135417 | 0.134115 | -0.001302 |
| gated | broad | 5 | 512 | 0.135417 | 0.134766 | -0.000651 |
| gated | broad | 10 | 512 | 0.135417 | 0.134766 | -0.000651 |
| gated | broad | 20 | 512 | 0.135417 | 0.135417 | 0.000000 |

The ablation results show no positive accuracy gain across the tested relation sets, top-k values, and fusion modes. Weighted fusion is consistently slightly negative, while gated fusion is safer and can preserve the baseline in the top-k 20 settings.

## Appendix D: Reproduction Commands

The following commands summarise the main reproduction workflow. Exact local paths depend on where OK-VQA, COCO images, and ConceptNet data are stored.

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset and vocabulary checks

```bash
python -m scripts.data_check_okvqa --config configs/baseline.yaml --max_image_checks 200
python -m scripts.build_answer_vocab --config configs/baseline.yaml --top_n 10000
python -m scripts.vocab_coverage --config configs/baseline.yaml --vocab data/processed/okvqa/answer_vocab.json
```

### Baseline training command

```bash
python -m src.train_baseline \
  --config configs/baseline_train_v4_suggested.yaml \
  --tag baseline_v4_pw20_epX_fullval
```

### Fusion training command

```bash
python -m src.train_fusion \
  --config configs/fusion_train_v3_topn.yaml \
  --tag <RUN_TAG>
```

### Random-slice control command

```bash
python -m src.train_fusion \
  --config configs/fusion_train_random_control.yaml \
  --tag m6_random_slice_weighted_topn20_fullval
```

### Fusion evaluation command

```bash
python -m scripts.eval_fusion_run \
  --config configs/fusion_train_v3_topn.yaml \
  --fusion_run_dir experiments/runs/<FUSION_RUN_ID> \
  --tag eval_<FUSION_RUN_ID>
```

### Run summary command

```bash
python -m scripts.summarize_runs \
  --runs_dir experiments/runs \
  --out_dir reports/final_results
```

## Appendix E: Configuration Summary

| Configuration | Purpose |
|---|---|
| `configs/baseline.yaml` | Dataset and baseline paths. |
| `configs/baseline_train_v4_suggested.yaml` | Baseline training configuration. |
| `configs/fusion_train_v3_topn.yaml` | Top-N fusion training/evaluation configuration. |
| `configs/fusion_train_random_control.yaml` | Random-slice control configuration. |
| Ablation configs | Relation-set, top-k, and fusion-mode diagnostic runs. |

Important configuration variables include:

| Variable | Purpose |
|---|---|
| `kg.hop_depth` | Controls graph expansion depth. |
| `kg.top_k` | Controls number of selected facts. |
| `kg.relation_set` | Selects strict or broad ConceptNet relation filter. |
| `kg.neighbor_limit` | Bounds ConceptNet neighbour expansion. |
| `kg.max_entities` | Limits extracted question entities. |
| `kg.random_slice` | Enables deterministic unrelated KG control slices. |
| `fusion.mode` | Selects weighted or gated fusion. |
| `fusion.topn_rerank` | Restricts KG influence to top-N baseline candidates. |
| `embed.temperature` | Scales KG answer-similarity logits. |

## Appendix F: Risk Register Summary

| Risk | Impact | Mitigation |
|---|---|---|
| Baseline instability | Fusion comparisons become invalid | Freeze a selected baseline for matched comparison. |
| Noisy KG retrieval | KG branch may harm accuracy | Use bounded slicing, relation filtering, top-k selection, top-N fusion, and gated fusion. |
| Cache contamination | Experiments may reuse incompatible slices | Include slice-affecting configuration values in the cache hash. |
| Weak attribution | Improvements or degradations may be misinterpreted | Add random-slice control. |
| Overclaiming | Dissertation may imply unsupported improvement | State clearly that KG did not improve accuracy in this implementation. |
| Compute limits | Full experiment matrix may be expensive | Use full validation for headline runs and 512-example subset for diagnostic ablations. |
| Dataset path issues | Reproduction may fail on a new machine | Document expected data layout and setup commands. |
| KG coverage gaps | ConceptNet may not include useful facts for all questions | Discuss as a limitation and future-work area. |

## Appendix G: Evidence Traceability Matrix

| Claim in dissertation | Evidence source |
|---|---|
| The final frozen baseline achieved 0.163892 VQA-soft accuracy | `dissertation_pack/experiments/runs/BASELINE_FREEZE_20260312_1456/metrics.json` |
| Naive weighted KG fusion degraded performance | `reports/final_results/final_results_summary.md` |
| Top-N constrained weighted fusion reduced harm but did not improve accuracy | `reports/final_results/final_results_summary.md` |
| Gated fusion preserved the frozen baseline | `reports/final_results/final_results_summary.md` |
| Random-slice control did not create a spurious gain | `reports/final_results/random_slice_control.md` |
| The KG branch uses bounded ConceptNet slicing | `src/kg/slice_builder.py` |
| The random-slice flag is part of slice configuration | `src/kg/slice_builder.py`, `configs/fusion_train_random_control.yaml` |
| KG fact encoding uses sentence embeddings and answer-vocabulary similarity | `src/kg/knowledge_encoder.py` |
| Evaluation compares baseline and fused predictions | `scripts/eval_fusion_run.py` |
| Error analysis is supported by prediction dumps | `scripts/error_analysis_dump.py` |

## Appendix H: Demonstration Artefacts

The demonstration should show:

1. The project README and research question.
2. The system architecture.
3. The final frozen baseline metrics.
4. The KG slice builder and random-slice flag.
5. The knowledge encoder and fusion/evaluation pipeline.
6. The final results summary.
7. The random-slice control.
8. The final conclusion that KG augmentation did not improve this implementation.

Suggested demo command file:

```bash
#!/usr/bin/env bash

echo "=== Final Results Summary ==="
sed -n '1,160p' reports/final_results/final_results_summary.md

echo "=== Random Slice Control ==="
sed -n '1,120p' reports/final_results/random_slice_control.md

echo "=== Frozen Baseline Metrics ==="
cat dissertation_pack/experiments/runs/BASELINE_FREEZE_20260312_1456/metrics.json
```

## Appendix I: Final Submission Checklist

Before submission, confirm:

- [ ] Declaration form is first page of final report.
- [ ] Student number appears on the front page.
- [ ] GitHub/source-code access is included in the report.
- [ ] Demo video link or uploaded MP4 is included.
- [ ] README contains final baseline and result summary.
- [ ] Dissertation does not claim KG improved accuracy.
- [ ] Final results match `reports/final_results/final_results_summary.md`.
- [ ] Random-slice control is described accurately.
- [ ] All major claims are linked to run IDs, configs, scripts, or reports.
- [ ] No submission-facing TODO markers remain.
- [ ] Final PDF opens correctly.
- [ ] Video link works from a private/incognito browser.
