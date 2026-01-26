# OK-VQA Knowledge-Augmented VQA (Project Scaffold)

This repo scaffold is for:
- Baseline OK-VQA VQA model (ViLT) + evaluation
- Task-specific ConceptNet slice + knowledge branch
- Late fusion + ablations
- Reproducible run logging (`experiments/runs/<run_id>/`)

## Folder layout (minimum viable)
- `configs/` experiment configs (YAML)
- `docs/` project documentation (contract, protocol, metrics, risks, etc.)
- `scripts/` entrypoints (smoke test now; train/eval later)
- `src/` library code (datasets, models, KG, fusion, eval)
- `experiments/runs/` run outputs (gitignored)
- `data/` datasets + caches (gitignored)

## Quick start (smoke test)
1) Create a virtual environment and install requirements:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2) Make sure `run_metadata.py` is placed at the repo root (next to this README).

3) Run the smoke test (creates a run folder + writes run.json + metrics.json):
```bash
python -m scripts.smoke_test --config configs/baseline.yaml --tag m1_smoke
```

Output will be in:
- `experiments/runs/<run_id>/run.json`
- `experiments/runs/<run_id>/metrics.json`
- `experiments/runs/<run_id>/notes.md`

## Data placement (later)
When you download OK-VQA / COCO, update `configs/baseline.yaml` with correct paths.

## Documentation
Place the docs you generated into `docs/`, for example:
- `project_contract.md`
- `task_definition.md`
- `baseline_choice.md`
- `data_protocol.md`
- `metrics.md`
- `architecture.md`
- `risks.md`
- `experiment_log_template.md`
