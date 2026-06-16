# Implementation

## 1. Chapter Purpose

This chapter describes the implementation of the knowledge-augmented Visual Question Answering system. The project implements a controlled OK-VQA pipeline that compares a frozen ViLT baseline against ConceptNet-augmented variants using bounded knowledge graph slicing and late fusion.

The implementation was designed to support the research question rather than to produce a state-of-the-art VQA model. The main implementation goals were modularity, reproducibility, controlled comparison, and traceable evaluation. Each major component was implemented separately so that the baseline, knowledge retrieval, knowledge encoding, fusion, and evaluation stages could be tested and discussed independently.

## 2. Repository Organisation

The implementation is organised around the main stages of the VQA pipeline.

The `configs/` directory stores experiment configurations. These define baseline training settings, KG slice settings, fusion modes, top-N reranking settings, and random-slice control settings.

The `src/` directory contains the main system code. It includes dataset loading, baseline model logic, ConceptNet knowledge graph utilities, knowledge encoding, and fusion training.

The `scripts/` directory contains command-line tools for dataset checking, answer vocabulary construction, slice building, fusion evaluation, run summarisation, and error analysis.

The `reports/` and `reports/final_results/` directories contain result summaries and final evaluation artefacts.

The `dissertation_pack/experiments/runs/` directory stores selected run metrics used for dissertation reporting.

This structure allows a marker to trace dissertation claims back to source files, configurations, run IDs, and metrics.

## 3. Dataset Loading and Preprocessing

The system uses OK-VQA as the main dataset. OK-VQA provides image-question pairs and human answer annotations for visual questions that often require external knowledge.

The dataset loader is responsible for reading OK-VQA question files, reading annotation files, linking each question to a COCO image ID, and returning the fields needed by the baseline and evaluation pipeline. Each example must provide the question text, image identifier, question identifier, image input, and answer annotations.

The dataset stage is important because every later component depends on consistent identifiers. The KG slice cache, baseline predictions, fused predictions, and evaluation outputs all rely on the same question and image IDs. If these identifiers were inconsistent, baseline and KG predictions could be matched to the wrong examples.

The project also includes data checking commands to confirm that configured dataset paths and image files are available before training or evaluation.

## 4. Answer Vocabulary Construction

The baseline and fusion systems use a fixed answer vocabulary. This converts OK-VQA into a classification-style task where the model predicts scores over a predefined set of answer candidates.

The answer vocabulary was built from training annotations and limited to 10,000 answers. This design keeps the answer space computationally manageable while covering common answer forms in the dataset.

The fixed vocabulary is also necessary for late fusion. The baseline branch and the KG branch must both produce scores over the same answer index space. Without a shared answer vocabulary, their logits could not be combined directly.

This design has an important limitation. Some semantically reasonable answers may not be represented in the vocabulary, and VQA-style scoring still depends on matching short human answer strings. This limitation is discussed in the testing and evaluation chapter.

## 5. Baseline VQA Model

The baseline branch uses a ViLT-based classifier over the fixed answer vocabulary. ViLT is suitable for this project because it provides an established transformer-based image-question architecture while remaining practical for a local MSc implementation.

The final frozen baseline used for matched comparison is `BASELINE_FREEZE_20260312_1456`. Its validation VQA-soft accuracy is 0.163892 on 5046 OK-VQA validation examples.

Freezing the baseline was an important implementation decision. The research question asks whether the KG branch improves over the baseline. If the baseline continued changing during fusion experiments, then any result could be caused by changes in the baseline rather than by the knowledge augmentation. The implementation therefore treats the baseline as the stable comparator for all full-validation fusion runs.

The baseline implementation produces answer logits over the shared vocabulary. These logits are used directly for baseline-only evaluation and are also passed into the late-fusion stage for KG-augmented evaluation.

## 6. ConceptNet Knowledge Store

The knowledge branch uses ConceptNet as the external commonsense knowledge source. ConceptNet represents commonsense knowledge as relations between concepts, such as object properties, uses, locations, and associations.

For this project, ConceptNet is accessed through a local queryable store rather than repeatedly querying an external service. This improves reproducibility and reduces dependency on network access during experiments.

The ConceptNet store supports neighbour lookup for extracted seed concepts. It returns candidate edges with relation labels, neighbouring concepts, edge weights, and surface text. These neighbours are then filtered and ranked by the slice builder.

Using ConceptNet has both advantages and limitations. It is transparent and inspectable, which makes it suitable for a system-development dissertation. However, it is also noisy and often generic, which means retrieved facts may not align with the short answer expected by OK-VQA.

## 7. Entity Extraction

The KG branch begins by extracting candidate entities from the question text. These entities are used as seed concepts for ConceptNet lookup.

The entity extraction design is intentionally simple and reproducible. It uses question text rather than generated captions or object detectors. This keeps the pipeline bounded and easier to debug, but it also creates a limitation: visually important concepts that are not mentioned in the question may be missed.

For example, if a question refers indirectly to an object visible in the image, question-only entity extraction may not retrieve the most useful ConceptNet concepts. This limitation helps explain why the KG branch did not improve final accuracy.

## 8. Bounded KG Slice Builder

The slice builder constructs a task-specific ConceptNet slice for each image-question pair. A slice contains extracted entities, selected ConceptNet facts, scores, and statistics.

The slice builder is controlled by a `SliceConfig` object. The configuration includes hop depth, top-k fact limit, relation set, minimum edge weight, neighbour limit, maximum entities, maximum n-gram length, scorer version, and a random-slice control flag.

The implementation deliberately avoids unrestricted graph expansion. ConceptNet neighbourhoods can grow quickly and contain many weakly related facts. Bounded slicing makes the system more reproducible, auditable, and suitable for ablation.

The slice builder supports one-hop retrieval and optional two-hop expansion. For each seed concept, neighbouring ConceptNet edges are retrieved using the selected relation set and minimum weight threshold. Candidate facts are scored using a combination of ConceptNet edge weight and lexical overlap with the question. The highest-scoring facts are kept up to the configured top-k limit.

This approach provides a practical way to select a small amount of external knowledge for each question. However, the evaluation results show that bounded retrieval alone was not sufficient to produce useful answer evidence.

## 9. Slice Caching and Determinism

KG slices are cached to avoid expensive repeated ConceptNet retrieval and fact scoring. The cache key includes the question ID, image ID, and a hash of the slice-affecting configuration.

This is important for experiment validity. If the cache did not include configuration values such as top-k, relation set, hop depth, or random-slice mode, then different ablation settings could accidentally reuse incompatible slices. That would make reported comparisons unreliable.

The implementation includes the random-slice flag in the configuration hash so that random-control runs use separate cached slices from task-specific KG runs.

The random-slice mode is deterministic per question/image pair. This means the random control can be rerun consistently while still remaining unrelated to the question content.

## 10. Knowledge Encoding

After KG slices are built, the selected facts must be converted into a form that can influence answer prediction.

The knowledge encoder converts facts into text strings, embeds them using a sentence embedding model, pools the fact embeddings using fact scores as weights, and compares the resulting knowledge embedding with cached answer embeddings.

The implementation produces two outputs: a pooled KG embedding and KG logits over the answer vocabulary. The KG logits are computed by comparing the pooled knowledge embedding with answer embeddings using cosine similarity and applying a temperature scale. The code also caches answer embeddings for the selected embedding model and answer vocabulary, which improves efficiency and reproducibility.

This design allows the KG branch to produce a signal in the same answer space as the baseline branch. However, it also creates a key challenge: semantic similarity between a fact embedding and an answer embedding may not correspond to the correct OK-VQA answer. This is one likely reason why KG fusion did not improve final accuracy.

## 11. Late Fusion

The late-fusion implementation combines baseline logits with KG-derived logits.

Late fusion was chosen because it preserves modularity. The baseline branch can remain frozen, while the KG branch and fusion strategy can be changed independently. This makes the system suitable for controlled ablation.

The implementation supports weighted fusion and gated fusion.

Weighted fusion adds a scaled KG signal to the baseline logits. This directly tests whether the KG branch provides useful answer evidence. In the final full-validation results, naive weighted fusion substantially degraded accuracy, suggesting that the KG signal was noisy or poorly calibrated when applied broadly.

Gated fusion learns to regulate the KG contribution. In the final full-validation results, gated fusion preserved baseline performance. This should be interpreted as behavioural evidence that the gated configuration neutralised harmful KG influence in aggregate, not as direct proof of individual gate values unless those gate values are separately analysed.

The implementation also supports top-N constrained fusion. In this setting, KG influence is restricted to the baseline model’s top-N answer candidates. This was added to reduce the risk that KG evidence would perturb the entire answer distribution. Top-N constrained weighted fusion reduced the large degradation seen in naive weighted fusion, but it still did not improve over the frozen baseline.

## 12. Fusion Training and Evaluation

Fusion runs are configured through YAML files. These files define the baseline checkpoint, ConceptNet database path, KG slice parameters, embedding model, temperature, fusion mode, top-N reranking setting, training seed, batch size, and validation limits.

During fusion training, the baseline model is loaded and frozen. The KG branch builds or loads slices for each example, encodes selected facts, and produces KG logits. The fusion module combines baseline and KG logits, and the run logs metrics such as validation loss, baseline accuracy, fused accuracy, delta, slice configuration, and runtime.

The evaluation scripts compare saved fusion runs against the frozen baseline. This supports matched comparison and allows the final dissertation to report baseline and fused performance side by side.

The final full-validation runs showed that the implemented KG branch did not improve OK-VQA validation accuracy. This result is negative but informative because it was produced through a controlled implementation and evaluation pipeline.

## 13. Random-Slice Control Implementation

The random-slice control was implemented to test attribution. In random-slice mode, the KG branch samples unrelated ConceptNet concepts deterministically for each question/image pair. The rest of the KG encoding and fusion process remains the same.

This control asks whether the KG branch can create an artificial improvement simply because an extra branch, trainable fusion parameters, or top-N reranking mechanism is present.

The final random-slice control did not improve over the frozen baseline. This supports the interpretation that the system was not receiving a spurious boost from unrelated KG evidence.

## 14. Error Analysis Tooling

The project includes error-analysis tooling to inspect baseline and fused predictions. The error-analysis script can dump prediction records, summarise improved, worsened, and unchanged cases, and include previews of KG facts.

This tooling is important because quantitative accuracy alone does not explain why the KG branch failed to improve performance. Error analysis allows the dissertation to discuss likely failure modes such as weak entity extraction, generic ConceptNet facts, answer-vocabulary mismatch, and poor calibration of KG logits.

The error-analysis tooling should be treated as diagnostic support rather than as proof that every failure has been manually explained.

## 15. Run Logging and Result Packaging

A major implementation requirement was that final claims should be traceable. The project therefore stores run outputs, metrics, and summaries in repository artefacts.

The key final reporting files are:

- `reports/final_results/final_results_summary.md`;
- `reports/final_results/random_slice_control.md`;
- `reports/final_results/run_summary.md`;
- `reports/final_results/run_summary.csv`;
- selected metrics files under `dissertation_pack/experiments/runs/`.

This packaging allows the dissertation to refer to exact run IDs and result values. It also reduces the risk of stale or contradictory numbers appearing in the final report.

## 16. Implementation Challenges

Several implementation challenges shaped the final system.

The first challenge was baseline reliability. Early baseline runs performed poorly and were retained as development history. A later frozen baseline was selected for final matched comparison.

The second challenge was KG noise. ConceptNet retrieval often produces facts that are broadly related but not answer-discriminative. This required bounded slicing, relation filtering, top-k selection, and later gated/top-N fusion.

The third challenge was answer-space alignment. The KG encoder maps fact embeddings to answer logits using semantic similarity with answer embeddings. This does not guarantee that relevant facts will increase the correct short answer expected by the VQA metric.

The fourth challenge was reproducibility. Caches and run outputs needed to be linked to configurations so that ablation results could be trusted.

The fifth challenge was interpretation. The final result was negative, so the implementation had to support diagnostic claims rather than only reporting headline accuracy.

## 17. Implementation Limitations

The implementation has several limitations.

The baseline is a local project baseline rather than a state-of-the-art OK-VQA model.

The KG branch uses question-derived entities rather than visual object detections or generated captions. This limits the ability to retrieve facts about image content that is not explicitly named in the question.

ConceptNet facts are generic and may not align with the specific answer required by OK-VQA.

The knowledge encoder uses embedding similarity between fact text and answer strings. This is simple and auditable, but it may be too weak to support reliable answer correction.

The fusion mechanisms can reduce harm, but they cannot create useful gains if the KG signal itself is not sufficiently relevant or calibrated.

These limitations are consistent with the final evaluation results.

## 18. Summary

The implementation produced a modular KG-augmented VQA system with a frozen ViLT baseline, bounded ConceptNet slicing, cached KG retrieval, knowledge encoding, weighted and gated late fusion, top-N constrained reranking, ablation support, random-slice control, and final result packaging.

The system did not demonstrate an accuracy improvement from KG augmentation. However, the implementation provides a defensible system-development contribution because it supports controlled comparison, reproducibility, ablation, attribution control, and diagnostic interpretation of a negative result.
