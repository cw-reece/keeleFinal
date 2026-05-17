# Knowledge-Augmented Visual Question Answering via Task-Specific Knowledge Graph Slicing and Late Fusion

**Student:** Christopher Ward Reece  
**Student number:** 24034374  
**Course:** MSc Computer Science  
**Module:** CSC40098 MSc Project  
**Project type:** System development-focused project  

## Project summary

This repository contains the source code, configuration files, experiment scripts, and documentation for an MSc final project on knowledge-augmented Visual Question Answering (VQA).

The project investigates whether bounded, task-specific ConceptNet knowledge graph augmentation, integrated through late fusion, can improve OK-VQA validation performance compared with a frozen ViLT baseline under matched experimental conditions.

The system builds a baseline VQA pipeline, constructs bounded ConceptNet knowledge graph slices for each image-question pair, converts retrieved facts into a knowledge-derived answer signal, and combines that signal with baseline model logits using late-fusion strategies. The project is evaluated using OK-VQA validation VQA-style soft accuracy, ablation studies, control experiments, and qualitative error analysis.

## Research question

**Does bounded, task-specific ConceptNet knowledge graph augmentation, integrated via late fusion, improve OK-VQA validation VQA-soft accuracy compared with a frozen ViLT baseline under matched conditions?**

## Project contribution

The contribution of this project is not a claim of state-of-the-art VQA performance. The contribution is a reproducible system-development study that implements and evaluates a controlled knowledge-augmented VQA pipeline.

The project contributes:

- an end-to-end OK-VQA baseline using a ViLT-based answer classifier;
- a bounded ConceptNet slicing pipeline using question-derived entities;
- deterministic slice caching keyed by question, image, and slice configuration;
- knowledge encoding and late-fusion mechanisms;
- weighted and gated fusion comparisons;
- random-slice/control support for attribution checks;
- reproducible run logging and evaluation scripts;
- qualitative error-analysis outputs showing when KG evidence helps, hurts, or is ignored.

## System overview

The system has two main branches.

The first branch is the baseline VQA model. It takes an OK-VQA image and question, processes them using a ViLT-compatible processor, and outputs answer logits over a fixed answer vocabulary.

The second branch is the knowledge branch. It extracts entities from the question, queries ConceptNet, builds a bounded task-specific knowledge slice, ranks candidate facts, encodes the selected facts, and produces a knowledge-derived answer signal.

The two branches are combined using late fusion. This keeps the baseline and knowledge components separable, which makes ablation and diagnostic evaluation easier.

## High-level architecture

```
OK-VQA image + question
        |
        |-----------------------------|
        |                             |
        v                             v
ViLT baseline branch            KG knowledge branch
        |                             |
baseline answer logits          entity extraction
                                      |
                                ConceptNet lookup
                                      |
                                bounded KG slice
                                      |
                                fact ranking / encoding
                                      |
                                KG answer signal
        |                             |
        |------------- late fusion ---|
                       |
                 final answer logits
                       |
                  VQA-soft scoring
```

## Repository structure

```
.
├── configs/                  # Experiment and model configuration files
├── data/                     # Local data paths, processed vocab, and cache directories
├── docs/                     # Project documentation, architecture, risks, results notes
├── experiments/              # Logged runs, metrics, configs, and checkpoints
├── reports/                  # QC logs, final result tables, error analysis outputs
├── scripts/                  # Data checks, evaluation scripts, analysis utilities
├── src/
│   ├── datasets/             # OK-VQA dataset loading
│   ├── eval/                 # Normalisation and VQA scoring
│   ├── fusion/               # Late-fusion modules
│   ├── kg/                   # ConceptNet store, entity extraction, slicing, encoding
│   ├── models/               # ViLT answer classifier
│   └── utils/                # Config and run utilities
├── templates/                # Dissertation and results-section templates
└── README.md
```

## Core components

### OK-VQA dataset loader

The dataset loader reads OK-VQA question and annotation files and links them to COCO images. It provides image, question text, answer list, question ID, and image ID fields used by the baseline and evaluation pipelines.

### Baseline VQA model

The baseline uses a ViLT-based classifier over a fixed answer vocabulary. The baseline is treated as the frozen comparison point for knowledge-augmented experiments.

The current report-grade baseline recorded in the repository is:

```
Run ID: 20260304_092423_baseline_v4_pw20_ep3_fullval
Backbone: dandelin/vilt-b32-finetuned-vqa
Answer vocabulary: 10,000 answers
Validation split: OK-VQA validation, 5046 examples
Metric: VQA-style soft accuracy
Result: 0.056084
```

The run tag includes `ep3`, but the run metadata reports `epochs=2`. The metadata should be treated as the authoritative record.

### ConceptNet knowledge slicing

For each image-question pair, the system builds a bounded ConceptNet slice.

The slice configuration controls:

- hop depth;
- top-k fact limit;
- relation-set filtering;
- minimum ConceptNet edge weight;
- neighbour expansion limit;
- maximum extracted entities;
- n-gram extraction limit;
- scoring version;
- random-slice control flag.

Slice outputs include the extracted entities, selected facts, scores, and slice statistics. Slices are cached using a configuration hash to support deterministic reruns and avoid accidental cache contamination across ablation settings.

### Late fusion

The project evaluates late fusion because it keeps the baseline model and the knowledge branch separable. This allows direct comparison between baseline-only and knowledge-augmented predictions.

Implemented fusion modes include:

- weighted additive fusion;
- gated fusion;
- top-N reranking over the baseline answer space.

The purpose of fusion is not only to seek accuracy gains, but also to test whether external KG evidence can safely influence a frozen VQA baseline.

### Evaluation and diagnostics

The evaluation pipeline reports VQA-style soft accuracy for baseline and fused predictions.

The project also includes diagnostic tooling for:

- baseline versus fused prediction comparison;
- improved, worsened, and unchanged prediction counts;
- KG slice inspection;
- selected qualitative error-analysis cases;
- random-slice control comparisons;
- configuration-linked result logging.

## Data requirements

This project uses public datasets and resources:

- OK-VQA annotations;
- COCO image files used by OK-VQA;
- ConceptNet data processed into a local queryable store.

The dataset files are not included directly in this repository where they are too large or externally licensed. The expected local structure is configuration-driven, but a typical structure is:

```
data/
├── raw/
│   └── okvqa/
│       ├── annotations/
│       └── images/
│           ├── train2014/
│           └── val2014/
├── processed/
│   └── okvqa/
│       └── answer_vocab.json
└── cache/
    ├── okvqa/
    │   └── slices/
    └── embeddings/
```

## Environment setup

Create and activate a Python environment, then install the required dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using CUDA, install a PyTorch build compatible with the local CUDA version before running training or evaluation.

## Data preparation

Run the dataset integrity check:

```bash
python -m scripts.data_check_okvqa --config configs/baseline.yaml --max_image_checks 200
```

Build the answer vocabulary:

```bash
python -m scripts.build_answer_vocab --config configs/baseline.yaml --top_n 10000
```

Check vocabulary coverage:

```bash
python -m scripts.vocab_coverage \
  --config configs/baseline.yaml \
  --vocab data/processed/okvqa/answer_vocab.json
```

## Reproducing the baseline

Train the baseline using the selected baseline configuration:

```bash
python -m src.train_baseline \
  --config configs/baseline_train_v4_suggested.yaml \
  --tag baseline_v4_pw20_epX_fullval
```

The current report-grade baseline recorded in `docs/docs_baseline_results.md` is:

```
20260304_092423_baseline_v4_pw20_ep3_fullval
Validation VQA-soft accuracy: 0.056084
```

## Evaluating a fusion run

Evaluate a saved fusion run against the frozen baseline:

```bash
python -m scripts.eval_fusion_run \
  --config configs/fusion_train_v3_topn.yaml \
  --fusion_run_dir experiments/runs/<FUSION_RUN_ID> \
  --split val \
  --limit 0
```

Example:

```bash
python -m scripts.eval_fusion_run \
  --config configs/fusion_train_v3_topn.yaml \
  --fusion_run_dir experiments/runs/20260313_145031_m5_topn20_weighted_fullval \
  --split val \
  --limit 0
```

Replace the run directory with the final run selected for dissertation reporting.

## Error analysis

Generate qualitative error-analysis outputs:

```bash
python -m scripts.error_analysis_dump \
  --config configs/fusion_train_v3_topn.yaml \
  --fusion_run_dir experiments/runs/<FUSION_RUN_ID> \
  --split val \
  --limit 300 \
  --out_dir reports/error_analysis/<RUN_NAME>
```

This creates:

```
predictions.jsonl
summary.json
selected_cases.md
```

The selected cases are intended for dissertation discussion. They include the question, ground-truth answers, baseline prediction, fused prediction, VQA-soft scores, and KG facts used by the knowledge branch.

## QC and reproducibility

Run the QC script:

```bash
bash tools/qc_run_all.sh
```

Then review:

```
reports/qc_log.md
```

The QC log should be included as part of the final audit trail when writing the dissertation.

## Final results

The final dissertation should report results from configuration-linked run folders under:

```
experiments/runs/
```

The final results pack should be collected under:

```
reports/final_results/
```

Recommended final result artefacts:

```
reports/final_results/
├── baseline_results.md
├── fusion_comparison.csv
├── ablation_matrix.csv
├── slice_statistics.csv
├── random_slice_control.csv
├── error_analysis_summary.json
└── selected_cases.md
```

## Current headline result

Baseline:

```
Validation VQA-soft accuracy: 0.056084
Run ID: 20260304_092423_baseline_v4_pw20_ep3_fullval
```
The main full-validation fusion results were:

Naive weighted KG fusion: 0.110517, delta -0.053376
Top-20 weighted KG fusion: 0.163760, delta -0.000132
Top-20 gated KG fusion: 0.163892, delta 0.000000
Random-slice weighted control: 0.163760, delta -0.000132

The implemented bounded ConceptNet late-fusion branch did not improve OK-VQA validation accuracy over the frozen ViLT baseline. Naive weighted fusion substantially degraded performance, top-N constrained weighted fusion reduced the harm to a near-zero negative delta, gated fusion preserved baseline performance, and the random-slice control showed that unrelated KG evidence did not create a spurious gain.

Full result details are in:

reports/final_results/final_results_summary.md
reports/final_results/random_slice_control.md

The dissertation should state clearly whether the KG branch improved, matched, or degraded performance under matched conditions.

A null or negative result is still valid if the evaluation shows why: for example, noisy entity linking, generic ConceptNet neighbours, answer-space mismatch, weak KG-answer alignment, or gated fusion suppressing unreliable knowledge.

## Testing strategy
The project should be tested at several levels.

Unit-level checks:

- answer normalisation;
- VQA-soft scoring;
- entity extraction;
- relation filtering;
- cache-key generation;
- slice determinism.

Integration-level checks:

- OK-VQA loader to baseline model;
- entity extraction to ConceptNet retrieval;
- slice builder to knowledge encoder;
- baseline logits plus KG logits to fusion output;
- evaluation script against saved run directories.

System-level checks:

- reproduce the selected baseline run;
- evaluate final weighted fusion run;
- evaluate final gated fusion run;
- compare random-slice control;
- generate final error-analysis outputs.

## Known limitations

The project has several expected limitations:

- ConceptNet contains noisy and generic relations, so bounded slicing is necessary but not always sufficient.
- Entity extraction from question text can miss visually important entities or map to ambiguous concepts.
- Some OK-VQA questions require named-entity or encyclopaedic knowledge that ConceptNet may not cover well.
- Late fusion can only help if the KG-derived signal aligns with the answer vocabulary.
- Gated fusion may learn to suppress KG evidence if it is unreliable.
- Accuracy gains on OK-VQA may be small, so diagnostic analysis is essential.

## Academic framing

This is a system-development project. The project should therefore be judged by:

- clarity of objectives and requirements;
- quality of system design;
- working implementation;
- controlled testing and evaluation;
- reproducibility;
- honest discussion of limitations;
- quality of project management evidence.

The project does not claim that ConceptNet late fusion is universally superior to modern VQA systems. It tests a bounded and auditable form of knowledge augmentation under controlled conditions.

## Final submission checklist

Before submission, confirm that:

- the final report PDF includes the declaration form as the first page;
- the report front page includes the student number;
- the GitHub repository is accessible;
- the final README explains how to reproduce key results;
- the final report cites run IDs, configs, scripts, and result artefacts;
- the demo video link or MP4 is included;
- final result tables are present under `reports/final_results/`;
- qualitative error-analysis cases are available;
- limitations are stated honestly;
- no unsupported performance claims remain.

## Demo video outline

Recommended video structure:

1. State the problem and research question.
2. Show the high-level architecture.
3. Walk through the repository structure.
4. Show the baseline result and run log.
5. Show an example KG slice.
6. Show fusion evaluation output.
7. Show final result table.
8. Show one qualitative error-analysis case.
9. Conclude with the answer to the research question and limitations.

## References

The final dissertation should cite the relevant literature in Harvard style. Core sources include work on VQA, OK-VQA, ConceptNet, KRISP, graph-based reasoning, caption-based knowledge retrieval, and robustness in VQA.

This README is a project guide and reproduction aid. The formal literature review, full methodology, evaluation, discussion, and references are provided in the final dissertation.
