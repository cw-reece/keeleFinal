# Requirements and Design

## 1. Chapter Purpose

This chapter defines the functional and non-functional requirements for the project and explains the design of the implemented knowledge-augmented VQA system. The project is a system-development focused investigation into whether bounded ConceptNet knowledge graph augmentation, integrated through late fusion, improves OK-VQA validation performance compared with a frozen ViLT baseline.

The design priority was not to maximise raw model performance at any cost. The priority was to build a controlled, auditable, and reproducible system that could test the research question under matched conditions. This required a frozen baseline, a bounded knowledge retrieval pipeline, configurable fusion mechanisms, and evaluation scripts that logged results consistently.

## 2. System Scope

The system takes an OK-VQA image-question pair and predicts an answer from a fixed answer vocabulary. It compares two prediction settings:

1. A frozen ViLT-based baseline model.
2. A knowledge-augmented model where ConceptNet facts are retrieved, encoded, and fused with the baseline answer logits.

The system includes data loading, answer vocabulary construction, baseline evaluation, ConceptNet retrieval, KG slice construction, knowledge encoding, late fusion, ablation configuration, control experiments, and result logging.

The system does not attempt to build a state-of-the-art VQA model. It does not use large language model prompting, Wikidata retrieval, or end-to-end retraining of a large multimodal architecture. These were excluded to keep the project bounded and to preserve a clean comparison between the frozen baseline and the KG-augmented variants.

## 3. Functional Requirements

| ID | Requirement | Description | Implemented by | Evidence |
|---|---|---|---|---|
| FR1 | Load OK-VQA data | Load question, annotation, image ID, answer, and image path data for train and validation splits. | `src/datasets/okvqa.py` | Dataset checks and training/evaluation runs |
| FR2 | Build answer vocabulary | Construct a fixed answer vocabulary for classification-based VQA. | `scripts/build_answer_vocab.py` | `data/processed/okvqa/answer_vocab.json` |
| FR3 | Evaluate frozen baseline | Evaluate a ViLT-based baseline over the fixed answer vocabulary. | `src/train_baseline.py`, evaluation scripts | `BASELINE_FREEZE_20260312_1456` metrics |
| FR4 | Extract question entities | Extract candidate entities from question text for KG lookup. | `src/kg/entity_extract.py` | Slice samples and KG slice outputs |
| FR5 | Query ConceptNet | Retrieve neighbours and relations from a local ConceptNet store. | `src/kg/conceptnet_store.py` | KG slice construction |
| FR6 | Build bounded KG slices | Build task-specific KG slices bounded by hop depth, top-k, relation set, and neighbour limit. | `src/kg/slice_builder.py` | Slice stats and cached slice outputs |
| FR7 | Cache KG slices | Cache slices using a configuration-aware key to avoid recomputation and cache contamination. | `src/kg/cache.py`, `src/kg/slice_builder.py` | Config hash includes slice parameters |
| FR8 | Encode KG facts | Convert retrieved facts into a knowledge-derived answer signal. | `src/kg/knowledge_encoder.py` | Fusion training/evaluation runs |
| FR9 | Fuse baseline and KG outputs | Combine baseline logits and KG-derived logits using weighted or gated late fusion. | `src/fusion/late_fusion.py`, `src/train_fusion.py` | Weighted and gated fusion metrics |
| FR10 | Support top-N constrained fusion | Restrict KG influence to the baseline model’s top-N candidate answers. | `src/train_fusion.py`, `scripts/eval_fusion_run.py` | Top-20 and top-50 fusion runs |
| FR11 | Support random-slice control | Generate unrelated deterministic KG slices to test whether KG branch effects are spurious. | `SliceConfig.random_slice`, `fusion_train_random_control.yaml` | `random_slice_control.md` |
| FR12 | Log experimental results | Store metrics, configuration, run IDs, and evaluation outputs for traceability. | Run folders under `experiments/runs` and `dissertation_pack/experiments/runs` | `reports/final_results/final_results_summary.md` |
| FR13 | Produce diagnostic summaries | Summarise runs and compare baseline/fused performance. | `scripts/summarize_runs.py` | `reports/final_results/run_summary.md` |
| FR14 | Support error analysis | Dump baseline/fused predictions and KG facts for qualitative inspection. | `scripts/error_analysis_dump.py` | Error-analysis reports |

## 4. Non-Functional Requirements

| ID | Requirement | Description | Design response |
|---|---|---|---|
| NFR1 | Reproducibility | Experiments must be traceable to configs, run IDs, and metrics. | Run folders log metrics, configuration, seed, and settings. |
| NFR2 | Comparability | KG-augmented runs must be compared against the same frozen baseline. | The baseline checkpoint is fixed for fusion evaluation. |
| NFR3 | Determinism | KG slice generation and cache keys must be stable across runs. | Slice configuration is hashed and random slices are deterministic per question/image pair. |
| NFR4 | Bounded computation | KG retrieval must avoid unbounded graph expansion. | Hop depth, top-k, relation set, neighbour limit, and max entities are configurable. |
| NFR5 | Auditability | Retrieved KG evidence must be inspectable. | Slice objects store entities, facts, scores, and stats. |
| NFR6 | Modularity | Baseline, KG retrieval, encoding, fusion, and evaluation should be separable. | The system uses separate modules for datasets, models, KG, fusion, and evaluation. |
| NFR7 | Controlled evaluation | The system must support ablations and controls. | Configurable relation set, top-k, fusion mode, top-N rerank, and random-slice flag. |
| NFR8 | Robustness to noisy KG evidence | Fusion should not allow unreliable KG evidence to dominate unchecked. | Gated fusion and top-N constrained fusion were implemented. |
| NFR9 | Explainability | The dissertation must explain not only results, but why they occurred. | Error-analysis and slice inspection scripts support qualitative analysis. |
| NFR10 | Submission traceability | A marker should be able to connect report claims to repository artefacts. | Final summaries are stored under `reports/final_results/`. |

## 5. High-Level Architecture

The system is organised as a two-branch architecture.

The first branch is the frozen VQA baseline. It processes the image and question using a ViLT-compatible processor and produces answer logits over the fixed answer vocabulary.

The second branch is the KG knowledge branch. It extracts entities from the question, retrieves ConceptNet neighbours, builds a bounded task-specific knowledge slice, encodes the selected facts, and produces a KG-derived answer signal.

The outputs of the two branches are combined through late fusion. This keeps the KG branch separable from the baseline and allows the KG component to be evaluated without retraining the full vision-language model.

```text
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

## 6. Baseline Design

The baseline is a ViLT-based answer classifier. It maps each image-question pair to logits over a fixed answer vocabulary. The baseline is frozen for the main fusion comparisons so that the evaluation isolates the effect of the KG branch and fusion mechanism.

The baseline design supports the research question because it provides a stable comparator. If the baseline were changed between runs, it would be impossible to determine whether performance differences came from the knowledge branch or from baseline variation.

The final frozen baseline used for matched comparison is `BASELINE_FREEZE_20260312_1456`, with validation VQA-soft accuracy of 0.163892 on 5046 OK-VQA validation examples.

## 7. Knowledge Graph Slice Design

The KG branch uses ConceptNet as a commonsense knowledge source. Rather than injecting a large or unrestricted graph, the system builds a bounded slice for each question.

The slice configuration controls:

- hop depth;
- top-k fact count;
- relation set;
- minimum edge weight;
- neighbour expansion limit;
- maximum extracted entities;
- maximum n-gram length;
- scorer version;
- random-slice control flag.

This design was chosen because unrestricted graph retrieval would be difficult to audit and could introduce large amounts of irrelevant knowledge. Bounded slicing makes retrieval reproducible, inspectable, and suitable for ablation.

Each slice stores the question ID, image ID, question text, extracted entities, selected facts, and slice statistics. The slice cache key includes a hash of the slice-affecting configuration. This prevents experiments with different KG settings from accidentally reusing incompatible cached slices.

## 8. Random-Slice Control Design

The random-slice control was included to test attribution. In this mode, the KG branch retrieves unrelated ConceptNet concepts deterministically for each question/image pair. The random slice is unrelated to the question content but reproducible across runs.

This control helps answer an important validity question: if a KG-augmented model improves, is the improvement caused by useful retrieved knowledge, or merely by the presence of an extra branch, extra trainable parameters, or reranking effects?

In the final random-slice control, the random KG branch did not improve over the frozen baseline. This supports the conclusion that the KG branch did not create an artificial gain.

## 9. Fusion Design

Late fusion was chosen because it keeps the baseline and KG branch separable. This has three advantages.

First, it allows the baseline to remain frozen, preserving a clean matched comparison. Second, it allows the KG branch to be ablated without retraining the whole vision-language model. Third, it makes it possible to inspect how different fusion strategies behave when KG evidence is noisy.

The project implements weighted fusion and gated fusion.

Weighted fusion adds a scaled KG-derived signal to the baseline logits. This directly tests whether the KG branch provides useful answer evidence. However, the evaluation showed that naive weighted fusion can be harmful when the KG signal is noisy or poorly calibrated.

Gated fusion learns whether to suppress or allow the KG signal. This design is safer because the model can learn to ignore unreliable external evidence. In the final results, gated fusion preserved baseline performance but did not improve it.

Top-N constrained fusion was added to prevent the KG branch from perturbing the entire answer distribution. Instead, KG influence is restricted to the baseline model’s top-N candidate answers. This reduced the large degradation seen in naive weighted fusion, but did not create a positive accuracy gain.

## 10. Evaluation Design

The evaluation design was based on matched comparison. Each KG-augmented run reports:

- baseline validation accuracy;
- fused validation accuracy;
- delta between fused and baseline accuracy;
- fusion mode;
- top-N rerank setting;
- KG slice configuration;
- validation example count;
- random seed;
- runtime.

The main full-validation runs used 5046 OK-VQA validation examples. Smaller ablation runs used 512 examples and are treated as diagnostic rather than headline evidence.

The evaluation includes:

1. frozen baseline;
2. naive weighted fusion;
3. gated fusion;
4. top-N constrained weighted fusion;
5. top-N constrained gated fusion;
6. top-k and relation-set ablations;
7. random-slice control.

This evaluation design supports both performance measurement and explanation. It does not simply ask whether the KG branch improves accuracy; it also asks how different fusion constraints behave and whether any observed effect can be attributed to relevant KG content.

## 11. Traceability Matrix

| Requirement | Design component | Repo evidence | Evaluation evidence |
|---|---|---|---|
| Fixed baseline comparison | Frozen ViLT checkpoint | `BASELINE_FREEZE_20260312_1456` | Baseline accuracy 0.163892 |
| Bounded KG retrieval | `SliceConfig` and `build_slice` | `src/kg/slice_builder.py` | Slice config logged in metrics |
| Cache safety | Config hash includes KG settings | `config_hash()` | Separate configs generate separate slices |
| Fusion comparison | Weighted and gated fusion | `src/fusion/late_fusion.py` | Full-validation fusion table |
| Safer KG influence | Top-N constrained reranking | `apply_topn_rerank()` | Top-20/top-50 runs |
| Attribution control | Random-slice mode | `random_slice: true` config | Random-slice control result |
| Reproducibility | Run folders and summaries | `reports/final_results/` | Final results summary |
| Diagnostic explanation | Error-analysis tooling | `scripts/error_analysis_dump.py` | Qualitative case outputs |

## 12. Design Limitations

The design deliberately favours control and auditability over raw performance. This makes the project suitable for a system-development dissertation, but it also creates limitations.

The KG branch relies primarily on question-derived entities. If the question omits visually important objects, the system may build a slice around incomplete concepts.

ConceptNet facts may be generic or weakly related to the required answer. Even bounded retrieval does not guarantee semantic usefulness.

The knowledge encoder must align retrieved facts with a short answer vocabulary. A fact can be relevant in natural language but still fail to move probability toward the expected VQA answer.

Late fusion can only improve performance if the KG signal is relevant and calibrated. The final results suggest that the current KG signal was not reliable enough to improve the frozen baseline.

## 13. Summary

The system was designed to provide a controlled test of bounded ConceptNet augmentation for OK-VQA. The architecture separates the frozen ViLT baseline from the KG branch, uses bounded and cached KG slicing, supports weighted and gated late fusion, and logs results for reproducibility.

This design enabled a defensible evaluation even though the final result was negative. The system showed that naive KG fusion can harm performance, constrained fusion can reduce that harm, gated fusion can preserve baseline performance, and random-slice control can test whether the KG branch creates spurious gains.
