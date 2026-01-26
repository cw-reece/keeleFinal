# Architecture — Knowledge-Augmented OK-VQA via Task-Specific KG Slicing + Late Fusion

**Student:** Christopher Ward Reece (Chris)  
**Student number:** XXXXXXXX  
**Project title:** Knowledge-Augmented Visual Question Answering via Task-Specific Knowledge Graph Slicing and Late Fusion  
**Date:** 2026-01-12  
**Status:** Frozen at M1 (may evolve via Change Control)

## 1. Overview
The system answers OK-VQA questions using (1) a baseline vision-language model and (2) an external knowledge branch built from ConceptNet. For each (image, question), the knowledge branch extracts entities, builds a bounded KG slice, ranks facts for relevance, encodes the top facts, and produces a knowledge signal. A late fusion module combines baseline logits with the knowledge signal to produce the final answer. Evaluation compares baseline vs augmented under controlled ablations (hop depth, top-K, relation filtering, fusion strategy).

## 2. Component diagram (Mermaid)
```mermaid
flowchart LR
  A[OK-VQA Dataset Loader] --> B[Preprocess: ViLTProcessor]
  B --> C[Baseline VQA Model (ViLT)]
  C --> C1[Baseline Answer Logits]

  A --> D[Question Text]
  A --> E[Image ID / Image]
  D --> F[Entity Extraction]
  E --> G[Optional Visual Concepts]
  F --> H[Slice Builder (ConceptNet)]
  G --> H
  H --> I[Relation Filter + Pruning]
  I --> J[Fact Ranking / Retrieval (Top-K)]
  J --> K[Fact Encoder / Scorer]
  K --> K1[Knowledge Representation]

  C1 --> L[Late Fusion Module]
  K1 --> L
  L --> M[Final Answer Logits]
  M --> N[Prediction + Evaluation]
```

## 3. Responsibilities and interfaces

### 3.1 Dataset loader (`src/datasets/okvqa.py`)
Outputs per sample:
- `image`
- `question_text`
- `answers` (raw list)
- `answer_target` (class index for classification baseline)
- `question_id`, `image_id`

### 3.2 Baseline model (`src/models/baseline_vilt.py`)
- Input: processed image + tokenized question
- Output: answer logits over fixed vocab

### 3.3 Entity extraction (`src/kg/entity_extract.py`)
- Input: `question_text` (required), optional visual concepts
- Output: canonical entity strings (normalized)

### 3.4 ConceptNet store (`src/kg/conceptnet_loader.py`)
- Input: ConceptNet files
- Output: queryable index for neighbor/edge lookup

### 3.5 Slice builder (`src/kg/slice_builder.py`)
- Input: entities, hop depth, relation filters
- Output: bounded candidate facts/triples

### 3.6 Fact filtering & ranking (`src/kg/fact_ranker.py`)
- Input: candidate facts + query
- Output: top-K facts + slice statistics

### 3.7 Knowledge encoder (`src/kg/knowledge_encoder.py`)
- Input: top-K fact texts
- Output: pooled knowledge representation (vector)

### 3.8 Late fusion (`src/fusion/late_fusion.py`)
- Input: baseline logits + knowledge representation
- Output: fused logits
- Modes: weighted (primary), gating (ablation)

### 3.9 Evaluation (`src/eval/eval_okvqa.py`)
- Input: predictions + ground truth
- Output: accuracy + diagnostics per `docs/metrics.md`

## 4. Determinism and caching
- Cache KG slices by `(image_id, question_id, slice_config_hash)` under `data/cache/okvqa/slices/`.
- All slice-affecting knobs must be in the cache key (hop depth, top-K, relation set, entity extraction rules).
- Runs write `run.json` and `metrics.json` to `experiments/runs/<run_id>/`.

## 5. Ablation knobs
- `kg_enabled`: on/off
- `hop_depth`: 1 vs 2
- `top_k`: 5 / 10 / 20
- `relation_set`: strict vs broad
- `fusion`: weighted vs gating

## 6. Definition of Done
Architecture is implemented when:
- baseline train+eval works end-to-end
- KG slice builder produces cached slices + stats
- knowledge encoder outputs a vector per sample
- late fusion produces final predictions
- evaluation logs primary + diagnostic metrics
