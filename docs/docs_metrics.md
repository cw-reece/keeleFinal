# Metrics — What We Report Every Run

**Student:** Christopher Ward Reece (Chris)  
**Student number:** XXXXXXXX  
**Project title:** Knowledge-Augmented Visual Question Answering via Task-Specific Knowledge Graph Slicing and Late Fusion  
**Date:** 2026-01-12  
**Status:** Frozen for comparability (changes require Change Control)

## 1. Purpose
Defines the metrics that are computed and logged for every experiment run so results remain comparable across:
- baseline vs KG-augmented
- slice configurations
- fusion strategies
- ablations

## 2. Primary metric
### 2.1 OK-VQA accuracy (VQA-style accuracy)
Use the standard OK-VQA/VQA evaluation scoring from the official/selected evaluation script.

**Reporting rules**
- Use **validation** for development/ablations.
- Use **test** only for final reporting (if allowed).
- Always record split name, number of examples, and eval script version (commit hash).

**Required fields in metrics.json**
- `accuracy`
- `split`
- `n_examples`

## 3. Secondary diagnostic metrics (required)
These explain results and KG health.

### 3.1 Slice coverage
- `slice_nonempty_rate`: fraction of examples with >= 1 fact after filtering.

### 3.2 Slice size statistics
- `facts_per_slice_mean`
- `facts_per_slice_median`
- `facts_per_slice_p95`

### 3.3 Entity extraction statistics
- `entities_per_question_mean`
- `entities_per_question_median`
- `entity_extraction_empty_rate`

### 3.4 Filtering impact
- `facts_filtered_rate`: fraction of candidate facts removed by filters.

### 3.5 Latency / throughput
- `slice_build_ms_mean` (or retrieval time if precomputed)

### 3.6 Training stability
- `best_epoch`
- `train_loss_final`
- `val_loss_best` (if available)
- `seed`

## 4. Baseline vs augmented comparisons
Every augmented run must report:
- `delta_accuracy_vs_baseline` = `accuracy_aug - accuracy_baseline`
- `baseline_run_id_reference` = run ID of baseline used

If the baseline changes, label it “Baseline v2” and don’t mix comparisons.

## 5. Ablation grid (minimum required)
Record the following knobs in every run’s `metrics.json` under `params`:
- `kg_enabled`: true/false
- `hop_depth`: 1/2
- `top_k`: e.g., 5/10/20
- `relation_set`: strict/broad
- `fusion`: weighted/gating
- `seed`: integer

Example:
```json
"params": {
  "kg_enabled": true,
  "hop_depth": 2,
  "top_k": 10,
  "relation_set": "strict",
  "fusion": "weighted",
  "seed": 42
}
```

## 6. Run folder requirements
Each run folder must contain:
- `run.json` (metadata: commit hash, environment, config snapshot)
- `metrics.json`
- `notes.md`
- (recommended) `predictions.jsonl` for error analysis
- (if KG) `slice_stats.json` + a small `slice_samples/` set

## 7. Change control
Any change to primary metric definition, evaluation script/normalization, or split usage rules requires a change request and a new doc version.
