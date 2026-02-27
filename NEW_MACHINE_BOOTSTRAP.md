# New Machine Bootstrap Guide (Keele MSc OK‑VQA Repo)

This document is the “do-this-in-order” checklist to get this project running on a new computer after cloning the repo.  
It is designed for situations where the new machine is **not** on the same network, so datasets must be downloaded again.

---

## 0) System prerequisites (Ubuntu/Pop!_OS)

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv unzip wget
```

Recommended for large downloads (more reliable than wget in many cases):

```bash
sudo apt-get install -y aria2
```

---

## 1) Clone the repo

```bash
git clone <YOUR_REPO_URL> keeleFinal
cd keeleFinal
```

---

## 2) Create venv + install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sanity import check:

```bash
python -c "import yaml, PIL, torch, transformers; print('deps ok')"
```

GPU check (optional):

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## 3) Download datasets into expected folders

Create the expected directory structure:

```bash
mkdir -p data/raw/okvqa/annotations
mkdir -p data/raw/okvqa/images
```

### 3A) OK‑VQA JSON files (small)

These are the OK‑VQA questions + annotations (train/val). Download and unzip:

```bash
cd data/raw/okvqa/annotations

wget -c https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip
wget -c https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip
wget -c https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip
wget -c https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip

unzip -o mscoco_train2014_annotations.json.zip
unzip -o mscoco_val2014_annotations.json.zip
unzip -o OpenEnded_mscoco_train2014_questions.json.zip
unzip -o OpenEnded_mscoco_val2014_questions.json.zip
```

Return to images directory:

```bash
cd ../images
```

You should now have:

- `data/raw/okvqa/annotations/mscoco_train2014_annotations.json`
- `data/raw/okvqa/annotations/mscoco_val2014_annotations.json`
- `data/raw/okvqa/annotations/OpenEnded_mscoco_train2014_questions.json`
- `data/raw/okvqa/annotations/OpenEnded_mscoco_val2014_questions.json`

### 3B) COCO 2014 images (big)

Use **HTTP** endpoints (avoids common TLS/certificate issues):

#### Option 1: wget (simple)
```bash
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/train2014.zip
```

#### Option 2: aria2 (recommended for reliability/speed)
```bash
aria2c -x 16 -s 16 http://images.cocodataset.org/zips/val2014.zip
aria2c -x 16 -s 16 http://images.cocodataset.org/zips/train2014.zip
```

Unzip:

```bash
unzip -o val2014.zip
unzip -o train2014.zip
```

Return to repo root:

```bash
cd ../../../..
```

You should now have:

- `data/raw/okvqa/images/val2014/COCO_val2014_*.jpg`
- `data/raw/okvqa/images/train2014/COCO_train2014_*.jpg`

---

## 4) Verify dataset integrity (must pass)

This confirms:
- train/val question and annotation JSONs load
- image paths resolve

```bash
python -m scripts.data_check_okvqa --config configs/baseline.yaml --max_image_checks 200
```

Target output:
- train image exists rate ≈ **1.000**
- val image exists rate ≈ **1.000**

If either is 0.000, your COCO image folders are not where the config expects.

---

## 5) Build answer vocabulary (baseline classification)

This builds `data/processed/okvqa/answer_vocab.json` from the train split:

```bash
python -m scripts.build_answer_vocab --config configs/baseline.yaml
```

---

## 6) Run baseline training smoke test (proves pipeline works)

This runs a small baseline training job and logs outputs under `experiments/runs/`:

```bash
python -m src.train_baseline --config configs/baseline_train.yaml --tag baseline_new_machine_smoke
```

Expected outputs (new run folder under `experiments/runs/<run_id>/`):
- `run.json`
- `config.yaml`
- `metrics.json`
- `checkpoints/model.pt`

---

## 7) Where results go

All run artifacts live under:

- `experiments/runs/<run_id>/`

The baseline training run writes:
- `metrics.json` (including loss + validation VQA soft accuracy)
- checkpoint under `checkpoints/`

---

## 8) Common issues

### HTTPS certificate problems when downloading COCO
Use the HTTP links listed above.

### CUDA + DataLoader multiprocessing issues
The baseline trainer is written to be CUDA-safe. Keep `num_workers: 0` in config initially.

### Hugging Face “unauthenticated requests”
This is not fatal. It may slow downloads. You can optionally set a token:
- `export HF_TOKEN=...` (or login via HF CLI) if you want higher rate limits.

---

## 9) Minimal “I’m ready to work again” confirmation

If these three commands work, you are back in business:

```bash
python -m scripts.data_check_okvqa --config configs/baseline.yaml --max_image_checks 50
python -m scripts.build_answer_vocab --config configs/baseline.yaml
python -m src.train_baseline --config configs/baseline_train.yaml --tag baseline_ready
```
