# Poster Brief (Draft) — CSC40098 MSc Project

## Title + identity

**Title:** Knowledge-Augmented Visual Question Answering via Task-Specific Knowledge Graph Slicing and Late Fusion  
**Student:** Christopher Ward Reece (Chris)  
**Course code:** CSC40098 (MSc Computer Science Final Project)  
**Institution:** Keele University  
**Supervisor(s):** [ADD NAME(S)]  

## 1-sentence problem statement

Modern VQA models can “see” the image and read the question, but they often fail on OK-VQA because many questions require external world knowledge that isn’t present in the pixels or the training distribution.

## 1-sentence contribution (“what I built”)

I built a reproducible OK-VQA pipeline that (1) trains a strong baseline and (2) extracts a small, task-specific ConceptNet “slice” per question (cached + auditable), designed to support late-fusion experiments that test whether KG context improves VQA accuracy.

## Key results (numbers)

- **Best baseline (OK-VQA val VQA soft accuracy): 0.163892** using a 10k answer vocab, VQA-initialized ViLT encoder, and weighted BCE training (run: `BASELINE_FREEZE_20260312_1456`).
- **Answer vocab coverage:** 10k vocab gives **~96.5% validation coverage** (val answers covered by vocab).
- **KG slicing coverage:** **nonempty slices = 0.80** on 200 OK-VQA val examples (strict, hop=1, top_k=10).
- **Slice size and bounds:** for strict hop=1 top_k=10, **facts per slice mean = 7.455**, **median = 10**, **p95 = 10** (bounded exactly at top_k when nonempty).
- **Efficiency + reproducibility:** mean slice build time is **~1.69 ms** (strict top_k=10) and cache rerun shows **cache_hit_rate = 1.0**; **4 unit tests passed** validating bounds, relation filtering, determinism/caching, and config hashing.

## Method summary (6–10 bullets)

- Use **OK-VQA** as the benchmark task and **VQA-style soft accuracy** as the primary metric.
- Build a **top-N answer vocabulary (N=10,000)** from the training split to enable fast classification and improve coverage.
- Train a **ViLT-based baseline** initialized from a VQA-finetuned checkpoint to prevent collapse and speed convergence.
- Use a **soft multi-label target** per question (based on annotator agreement) and train with **BCEWithLogitsLoss** using `pos_weight=20` to handle extreme label imbalance.
- Ingest **ConceptNet assertions** into a **local SQLite index** for fast, deterministic neighbor lookup.
- Extract question entities using a **deterministic heuristic** (tokenization + stopword removal + n-grams) and normalize to ConceptNet-style concept strings.
- Build a bounded **KG slice per (image, question)** by retrieving neighbors, filtering relations (strict/broad), scoring candidates, and keeping **top_k** facts.
- **Cache slices** on disk by config-hash + (question_id, image_id) for reproducibility and speed; emit `slice_stats.json` and human-readable `slice_samples/` for audit.
- Add a **test suite** validating: slice bounds, strict relation enforcement, determinism via caching, and config-hash changes with knob changes.
- (Next milestone) Convert slices into a **knowledge signal** and combine with baseline logits using **late fusion** (weighted and gated strategies) to measure delta vs baseline.

## Limitations + next steps (3 bullets)

- **Limitations:** current entity extraction + fact scoring are heuristic/lexical, so slices can be noisy and may miss useful multi-hop facts; OK-VQA questions that need specific named entities may be under-served.
- **Limitations:** late-fusion module is the critical remaining piece for measuring “knowledge helps,” and results will depend on how knowledge is encoded/scored against the answer space.
- **Next steps:** implement two fusion strategies (weighted + gated), run a small ablation matrix (relations/top_k/hops), and report baseline vs augmented deltas with error analysis (where KG helps vs hurts).
