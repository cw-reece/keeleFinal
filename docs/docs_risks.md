# Risk Register — OK-VQA Knowledge-Augmented VQA (Initial)

**Student:** Christopher Ward Reece (Chris)  
**Student number:** XXXXXXXX  
**Project title:** Knowledge-Augmented Visual Question Answering via Task-Specific Knowledge Graph Slicing and Late Fusion  
**Date:** 2026-01-12  
**Status:** Initial (M1). Updated throughout project; changes logged in experiment notes.

---

## 1. Purpose
This risk register identifies the most likely ways the project can fail (technically, operationally, or academically) and defines mitigations and early warning signs. The goal is not to eliminate risk, but to detect it early and prevent schedule collapse.

**Rating scale**
- **Likelihood:** Low / Medium / High
- **Impact:** Low / Medium / High
- **Priority:** Derived from (Likelihood × Impact) informally; treat High/High as “drop everything”.

---

## 2. Risk table (initial)

### R1 — Entity extraction / linking is too noisy (KG slices irrelevant)
- **Likelihood:** High  
- **Impact:** High  
- **Early warning signs:**  
  - High `entity_extraction_empty_rate`  
  - Slice samples contain obviously wrong concepts (e.g., “bank” river vs finance)  
  - No improvement even when top-K is small and relations are strict
- **Mitigations:**  
  - Start simple: question-only noun phrases + lemmatization + stopword filtering  
  - Add ConceptNet-friendly normalization (underscores, lowercase, common aliases)  
  - Maintain a small manual audit set (~100 examples) and track “slice relevance rate” qualitatively  
  - Add a fallback: if no entities, disable KG branch (and gating should learn this)
- **Owner:** Chris  
- **When to act:** Immediately in M3 if relevance rate is poor

### R2 — ConceptNet retrieval is slow (training/eval becomes impractical)
- **Likelihood:** Medium  
- **Impact:** High  
- **Early warning signs:**  
  - Slice build time dominates batch time  
  - GPU idle while CPU builds slices  
  - Total experiment turnaround exceeds 1–2 hours for small runs
- **Mitigations:**  
  - Pre-index ConceptNet into a fast structure (sqlite + indices or adjacency dict)  
  - Aggressive caching by `(image_id, question_id, slice_config_hash)`  
  - Precompute slices for train/val once configs stabilize  
  - Separate slice generation from training (two-stage pipeline)
- **Owner:** Chris  
- **When to act:** As soon as average slice build time > ~50–100ms/sample (ballpark)

### R3 — KG adds noise and reduces accuracy (“knowledge hurts”)
- **Likelihood:** High  
- **Impact:** Medium–High  
- **Early warning signs:**  
  - Accuracy drops vs baseline when KG enabled  
  - Larger top-K makes it worse  
  - Broad relation set hurts more than strict
- **Mitigations:**  
  - Keep top-K small initially (e.g., 5–10)  
  - Use strict relation whitelist first  
  - Add gating fusion ablation (learn to ignore KG when unreliable)  
  - Log slice stats + examples where KG flips correct → wrong
- **Owner:** Chris  
- **When to act:** First augmented run in M5

### R4 — Baseline training unstable or too slow (can’t anchor comparisons)
- **Likelihood:** Medium  
- **Impact:** High  
- **Early warning signs:**  
  - Huge seed variance  
  - Training diverges / NaNs  
  - Baseline run takes days per iteration
- **Mitigations:**  
  - Freeze baseline early (ViLT) and start smaller (fewer epochs, smaller batch)  
  - Use mixed precision if stable  
  - Run 2–3 seeds only for final claims; one fixed seed for dev  
  - Log everything (commit/config/seed) so variance is visible
- **Owner:** Chris  
- **When to act:** M2 baseline milestone

### R5 — Evaluation mismatch / answer normalization bugs (numbers are meaningless)
- **Likelihood:** Medium  
- **Impact:** High  
- **Early warning signs:**  
  - Accuracy wildly different from expected baselines without explanation  
  - Small code tweaks change scores unexpectedly  
  - Manual checks show correct answers counted wrong (or vice versa)
- **Mitigations:**  
  - Use a single evaluation script for all runs  
  - Centralize normalization in one function  
  - Add unit tests with known normalization cases (articles, punctuation, numbers)  
  - Log a small sample of (prediction, normalized prediction, gold) for sanity
- **Owner:** Chris  
- **When to act:** Before trusting any baseline table

### R6 — Slice cache contamination (wrong slices reused across configs)
- **Likelihood:** Medium  
- **Impact:** Medium–High  
- **Early warning signs:**  
  - Changing hop/top-K doesn’t change slice stats  
  - Two different configs produce identical slice outputs unexpectedly
- **Mitigations:**  
  - Include all slice-affecting params in `slice_config_hash`  
  - Store a `slice_config.json` alongside cached slices  
  - Never overwrite cache files; write new key each time  
  - Add a cache sanity check (log hash + params)
- **Owner:** Chris  
- **When to act:** When caching is first introduced (M3)

### R7 — Dataset acquisition / path issues (blocked on data access)
- **Likelihood:** Medium  
- **Impact:** Medium  
- **Early warning signs:**  
  - Missing images or annotations  
  - COCO image IDs don’t match OK-VQA references  
  - File structure differs from loader assumptions
- **Mitigations:**  
  - Add data integrity script (counts, missing files, ID mismatches)  
  - Keep loader robust: config-driven paths; clear errors  
  - Document exact filenames and sources in `docs/data_protocol.md`
- **Owner:** Chris  
- **When to act:** Start of M2

### R8 — Compute constraints prevent ablation grid completion
- **Likelihood:** Medium  
- **Impact:** Medium–High  
- **Early warning signs:**  
  - One run takes > 12 hours  
  - Cutting ablations to “save time” without recording it
- **Mitigations:**  
  - Stage ablations: quick low-epoch directional runs first  
  - Reduce grid size intelligently (keep knobs that answer the research question)  
  - Precompute slices/embeddings where possible  
  - Track an “ablation budget” (planned vs completed)
- **Owner:** Chris  
- **When to act:** Mid M5

### R9 — ConceptNet coverage gaps / domain mismatch for OK-VQA questions
- **Likelihood:** Medium  
- **Impact:** Medium  
- **Early warning signs:**  
  - High empty-slice rate even when entity extraction seems OK  
  - Retrieved facts are generic (“RelatedTo”) and unhelpful
- **Mitigations:**  
  - Tune relation whitelist toward commonsense relations (UsedFor, CapableOf, HasProperty, etc.)  
  - Increase hop depth cautiously (1→2) only if relevance remains good  
  - Consider a fallback: lexical retrieval over ConceptNet edge text
- **Owner:** Chris  
- **When to act:** M3–M4

### R10 — Fusion design leaks comparability (baseline no longer “fixed”)
- **Likelihood:** Low–Medium  
- **Impact:** High  
- **Early warning signs:**  
  - Baseline trained differently when KG enabled  
  - Different preprocessing/vocab between runs  
  - “Convenient tweaks” creep in during debugging
- **Mitigations:**  
  - Keep baseline training regime identical across comparisons  
  - Enforce config snapshots + commit hash for every run  
  - If baseline changes, label as Baseline v2 and restart comparison series
- **Owner:** Chris  
- **When to act:** Throughout (esp. M4–M5)

### R11 — Reproducibility gaps (can’t rerun results later)
- **Likelihood:** Medium  
- **Impact:** High  
- **Early warning signs:**  
  - “I don’t remember which config produced that table”  
  - Runs lack commit hash/environment info  
  - Results only exist as screenshots
- **Mitigations:**  
  - Use run folders (`run.json`, `metrics.json`, config snapshot) for every run  
  - Store plots/tables in run folders  
  - Automate metadata capture (`run_metadata.py`)
- **Owner:** Chris  
- **When to act:** Immediately (M1–M2)

### R12 — Scope creep (Wikidata, bigger models, extra modules derail timeline)
- **Likelihood:** Medium  
- **Impact:** Medium–High  
- **Early warning signs:**  
  - “Just one more improvement” becomes a week  
  - Milestone dates slip without new evidence
- **Mitigations:**  
  - Treat `docs/project_contract.md` as law  
  - Use change control notes (`docs/changes/CHG_###.md`)  
  - Prioritize: baseline → KG slice → fusion → ablations → analysis
- **Owner:** Chris  
- **When to act:** Any time a new idea appears

---

## 3. High-priority actions (do these early)
1. Build a small manual slice audit set (≈100 samples) and use it to judge entity extraction/slice relevance quickly.
2. Implement caching + config hashing correctly the first time.
3. Lock evaluation and normalization early and add a couple unit tests.

---

## 4. Update cadence
- Review and update this register at least every **two-week progress report**.
- For any risk that becomes an issue, record the decision and outcome in the relevant run `notes.md`.
