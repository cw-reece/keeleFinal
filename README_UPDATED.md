# OK-VQA Knowledge-Augmented VQA (MSc Project)

**Student:** Christopher Ward Reece (Chris)  
**Project:** Knowledge-Augmented Visual Question Answering via Task-Specific Knowledge Graph Slicing and Late Fusion  
**Status:** Baseline pipeline operational; KG slicing and fusion in progress.

This repo implements:
- A reproducible **baseline VQA model** (ViLT backbone + answer-vocab classifier) on **OK-VQA**
- A pipeline for **ConceptNet ingestion + task-specific slicing** (Milestone 3)
- **Late fusion** experiments (Milestone 5)
- A reproducible run logging system (`experiments/runs/<run_id>/`)

---

## Repository layout

- `configs/`  
  YAML configs for runs (baseline training, KG slicing, fusion experiments).

- `docs/`  
  Project documentation (contract, task definition, data protocol, metrics, risks, etc.).

- `scripts/`  
  Utility scripts (dataset integrity checks, vocab building, etc.).

- `src/`  
  Core implementation: dataset loaders, baseline training, KG modules, eval utilities.

- `data/` *(gitignored)*  
  Raw datasets + processed artifacts + caches.

- `experiments/runs/` *(gitignored)*  
  Run outputs: metadata, metrics, checkpoints, plots/tables.

---

## Environment setup

### Option A: venv (recommended)
From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sanity check:
```bash
python -c "import yaml, PIL, torch, transformers; print('deps ok')"
```

### GPU notes
Training will run on CUDA automatically if available. Confirm:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## Data setup (OK-VQA + COCO 2014)

This repo expects:

```
data/raw/okvqa/
  annotations/
    mscoco_train2014_annotations.json
    mscoco_val2014_annotations.json
    OpenEnded_mscoco_train2014_questions.json
    OpenEnded_mscoco_val2014_questions.json
  images/
    train2014/
      COCO_train2014_*.jpg
    val2014/
      COCO_val2014_*.jpg
```

### 1) Create folders
```bash
mkdir -p data/raw/okvqa/annotations
mkdir -p data/raw/okvqa/images
```

### 2) OK-VQA annotations/questions
Place the 4 JSON files in:
`data/raw/okvqa/annotations/`

### 3) COCO 2014 images
Extract into:
- `data/raw/okvqa/images/train2014/`
- `data/raw/okvqa/images/val2014/`

### 4) Verify dataset integrity
```bash
python -m scripts.data_check_okvqa --config configs/baseline.yaml --max_image_checks 200
```

Target output:
- train image exists rate ≈ 1.000
- val image exists rate ≈ 1.000

---

## Answer vocabulary (classification baseline)

Build answer vocab from training set (top-N answers):
```bash
python -m scripts.build_answer_vocab --config configs/baseline.yaml
```

Output:
- `data/processed/okvqa/answer_vocab.json`

---

## Run logging / reproducibility

Every run creates a folder:
`experiments/runs/<run_id>/` containing:
- `run.json` (git commit, environment, command args, etc.)
- `config.yaml` (snapshotted config)
- `metrics.json`
- `checkpoints/` (if training)
- optional logs/plots/samples depending on the script

This is implemented by `run_metadata.py`.

---

## Baseline training (ViLT backbone + vocab head)

Baseline training config:
- `configs/baseline_train.yaml`

Run a small sanity training run:
```bash
python -m src.train_baseline --config configs/baseline_train.yaml --tag baseline_v1_ep1
```

Outputs:
- `experiments/runs/<run_id>/checkpoints/model.pt`
- `experiments/runs/<run_id>/metrics.json`

Notes:
- Initial runs often use `limit_train`/`limit_val` for speed.
- For report-grade baseline results, increase limits and epochs.

---

## Moving to a more powerful computer (recommended workflow)

Goal: keep **code in git**, keep **data and run outputs out of git**.

### What to copy
On the new machine:
1) Clone the repo
2) Recreate the Python environment (`requirements.txt`)
3) Copy these directories from the old machine (optional but useful):
- `data/raw/okvqa/` (datasets)  
- `data/processed/okvqa/answer_vocab.json`  
- `data/cache/` (if you’ve started KG slice caching)  
- `experiments/runs/<important_run_ids>/` (only the runs you want to keep)

### Suggested transfer method

**Option A: rsync over SSH**
```bash
rsync -av --progress data/raw/okvqa user@NEW_MACHINE:/path/to/repo/data/raw/
rsync -av --progress data/processed/okvqa user@NEW_MACHINE:/path/to/repo/data/processed/
rsync -av --progress experiments/runs/20260227_1230_baseline_v1_ep1 user@NEW_MACHINE:/path/to/repo/experiments/runs/
```

**Option B: external drive**
Copy the same folders as above.

### Verify on new machine
```bash
python -m scripts.data_check_okvqa --config configs/baseline.yaml --max_image_checks 50
python -m src.train_baseline --config configs/baseline_train.yaml --tag baseline_moved_smoke
```

---

## Project documentation
Key docs live under `docs/`:
- `project_contract.md`
- `task_definition.md`
- `baseline_choice.md`
- `data_protocol.md`
- `metrics.md`
- `architecture.md`
- `risks.md`
- `experiment_log_template.md`

---

## Known issues / troubleshooting

### COCO download HTTPS certificate errors
If HTTPS fails, use HTTP for COCO zip downloads:
- `http://images.cocodataset.org/zips/val2014.zip`
- `http://images.cocodataset.org/zips/train2014.zip`

### CUDA + DataLoader multiprocess issues
The baseline trainer uses a CUDA-safe pipeline. Keep `num_workers: 0` initially. Increase later only if stable.

---

## Next milestones (high-level)
- Milestone 3: ConceptNet ingestion + task-specific slicing (cached, bounded, auditable samples)
- Milestone 4: Knowledge module + tests/validation notes
- Milestone 5: Late fusion strategies + baseline/augmented comparison table
- Milestone 6: Automated experiment matrix + ablations + plots/tables + reproducibility checks
