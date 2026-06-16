# Methodology

## 1. Chapter Purpose

This chapter explains the development and evaluation methodology used in the project. The project was conducted as a system-development investigation rather than as a purely theoretical or benchmark-optimisation study. The central aim was to build a reproducible knowledge-augmented Visual Question Answering system and evaluate whether bounded ConceptNet knowledge graph augmentation, integrated through late fusion, improves OK-VQA validation performance compared with a frozen ViLT baseline.

The methodology was baseline-first, iterative, and experiment-driven. This was necessary because the research question depends on controlled comparison. The knowledge-augmented system can only be interpreted meaningfully if it is compared against a stable baseline under matched conditions.

## 2. Research Question

The fixed research question was:

> Does bounded, task-specific ConceptNet knowledge graph augmentation, integrated via late fusion, improve OK-VQA validation VQA-soft accuracy compared with a frozen ViLT baseline under matched conditions?

This question shaped the methodology in three ways.

First, the project required a working baseline VQA model before KG augmentation could be evaluated.

Second, the KG branch had to be modular so that it could be enabled, disabled, ablated, and controlled.

Third, evaluation had to compare baseline and fused predictions using the same validation examples and the same scoring method.

The project therefore prioritised matched comparison, reproducibility, and traceable experiment evidence over attempting to maximise absolute benchmark performance.

## 3. Methodological Approach

The project used an iterative system-development methodology with five main cycles:

1. Baseline establishment.
2. Knowledge graph pipeline construction.
3. Late-fusion integration.
4. Controlled evaluation and ablation.
5. Reporting, reproducibility, and dissertation packaging.

Each cycle produced a working artefact that could be tested before moving to the next stage. This reduced the risk of building a large integrated system without knowing which component caused failures.

The methodology was also experiment-driven. Design decisions were evaluated through logged runs, configuration files, metrics, and result summaries. This allowed the final dissertation to connect claims to concrete repository artefacts rather than relying on informal observations.

## 4. Cycle 1: Baseline Establishment

The first development cycle focused on establishing a ViLT-based OK-VQA baseline. This was essential because the KG-augmented system needed a stable comparator.

The baseline cycle included:

- loading OK-VQA question and annotation files;
- linking examples to COCO images;
- constructing a fixed answer vocabulary;
- training and evaluating a ViLT-based classifier;
- recording validation VQA-soft accuracy;
- freezing the selected baseline for later comparison.

Several baseline runs were treated as development history. The final frozen baseline used for matched comparison was `BASELINE_FREEZE_20260312_1456`, which achieved validation VQA-soft accuracy of 0.163892 on 5046 OK-VQA validation examples.

Freezing the baseline was a key methodological decision. Without a frozen baseline, changes in fused performance could be caused by baseline variation rather than the KG branch.

## 5. Cycle 2: Knowledge Graph Pipeline Construction

The second cycle built the ConceptNet knowledge pipeline. The purpose was to retrieve bounded, task-specific commonsense knowledge for each OK-VQA image-question pair.

This cycle included:

- creating or loading a local ConceptNet store;
- extracting candidate entities from question text;
- querying ConceptNet neighbours for those entities;
- filtering relations;
- ranking candidate facts;
- limiting retrieved facts using top-k selection;
- caching slices using configuration-aware keys.

The KG pipeline was deliberately bounded. ConceptNet can produce large and noisy neighbourhoods, so unbounded retrieval would make the system difficult to evaluate and explain. The slice configuration therefore included hop depth, top-k, relation set, minimum weight, neighbour limit, maximum entities, maximum n-gram length, scorer version, and a random-slice control flag.

This stage produced inspectable slice objects containing extracted entities, selected facts, scores, and slice statistics. This supported both reproducibility and qualitative error analysis.

## 6. Cycle 3: Late-Fusion Integration

The third cycle integrated the KG branch with the baseline using late fusion.

Late fusion was selected because it keeps the baseline and KG branch separable. This supports the research question because the baseline can remain frozen while KG-derived answer signals are added, constrained, gated, or disabled.

The project implemented three main fusion ideas:

- weighted fusion;
- gated fusion;
- top-N constrained reranking.

Weighted fusion directly adds a scaled KG signal to the baseline answer logits. This tests whether the KG branch contributes useful answer evidence.

Gated fusion learns whether the KG signal should be suppressed or allowed. This was included because external KG evidence may be noisy, and the system needed a safer mechanism than direct addition.

Top-N constrained fusion restricts KG influence to the baseline model’s most plausible answer candidates. This was included to reduce the risk of KG evidence perturbing the entire answer distribution.

The fusion cycle produced trainable and evaluable KG-augmented variants while preserving the frozen baseline for comparison.

## 7. Cycle 4: Controlled Evaluation and Ablation

The fourth cycle evaluated the system under matched conditions.

The primary metric was OK-VQA validation VQA-soft accuracy. Each fusion run reported:

- baseline validation accuracy;
- fused validation accuracy;
- delta between fused and baseline accuracy;
- fusion mode;
- top-N rerank setting;
- KG slice configuration;
- validation example count;
- seed;
- runtime.

The full-validation experiments used 5046 OK-VQA validation examples. These runs form the main evidence for the research question.

A smaller ablation matrix used 512 validation examples. These runs were used diagnostically to compare relation set, top-k value, and fusion mode. They are not treated as headline performance evidence.

A random-slice control was also included. This control generated unrelated but deterministic ConceptNet slices for each question/image pair. Its purpose was to test whether the KG branch could improve performance merely because an extra branch, extra parameters, or reranking mechanism was present.

The final evaluation showed that the KG branch did not improve the frozen baseline. Naive weighted fusion substantially degraded performance, top-N constrained weighted fusion reduced the harm to a near-zero negative delta, gated fusion preserved baseline performance, and the random-slice control did not create a spurious gain.

## 8. Cycle 5: Reporting and Reproducibility Packaging

The final cycle focused on turning experiment outputs into submission-ready evidence.

This included:

- cleaning the README so it identified the final frozen baseline and final result files;
- consolidating final metrics under `reports/final_results/`;
- writing a final results summary;
- documenting the random-slice control;
- drafting dissertation chapters;
- preparing a demonstration script;
- checking that claims matched run IDs and metrics.

This stage was necessary because a system-development dissertation is assessed not only on whether code exists, but also on whether the system can be understood, evaluated, and defended.

## 9. Quality Assurance

Quality assurance was built into the methodology through repeated checks rather than a single final test.

The project used several forms of QA:

- dataset integrity checks;
- vocabulary coverage checks;
- logged run configurations;
- frozen baseline comparison;
- full-validation evaluation;
- ablation matrix evaluation;
- random-slice control;
- result summary files;
- consistency checks across README, docs, reports, and dissertation drafts.

The project also used repository-level traceability. Major claims were connected to files such as metrics JSON files, final result summaries, random-control documentation, and evaluation scripts.

This helped reduce the risk of reporting stale or contradictory results.

## 10. Ethical Considerations

The project used public datasets and public knowledge resources. It did not collect personal data, involve human participants, or process sensitive user information.

The main ethical risk was representational rather than personal. VQA datasets and knowledge graphs can contain social, cultural, and dataset biases. The project does not claim that the model is fair, unbiased, or suitable for deployment. The system is evaluated as a research prototype for studying KG augmentation under controlled conditions.

The dissertation should therefore present the work as an experimental system-development project, not as a deployed decision-making tool.

## 11. Risk Management

Several project risks shaped the methodology.

The first risk was baseline instability. This was addressed by selecting and freezing a baseline before evaluating KG augmentation.

The second risk was noisy KG retrieval. This was addressed through bounded slicing, relation filtering, top-k selection, and later gated/top-N fusion.

The third risk was cache contamination. This was addressed by including slice-affecting configuration values in the cache hash.

The fourth risk was overclaiming. This was addressed by separating full-validation results from smaller diagnostic ablations and by stating clearly that the implemented KG branch did not improve accuracy.

The fifth risk was weak attribution. This was addressed through the random-slice control.

## 12. Methodological Limitations

The methodology has several limitations.

The baseline is not a state-of-the-art OK-VQA model. This limits the absolute benchmark value of the results, but it does not invalidate the matched comparison within this system.

The ablation matrix used a smaller 512-example validation subset. These results are useful for diagnosis but should not be treated as final headline performance.

The KG branch relies heavily on question-derived entities. If important visual concepts are not present in the question text, the retrieved ConceptNet slice may be incomplete or irrelevant.

The random-slice control tests whether unrelated KG evidence creates spurious improvement, but it does not directly measure whether task-specific slices are semantically high quality.

Despite these limitations, the methodology supports a defensible answer to the research question because it uses a frozen baseline, matched evaluation, ablations, and an attribution control.

## 13. Summary

The project followed a baseline-first, experiment-driven system-development methodology. This approach was appropriate because the research question required controlled comparison between a frozen ViLT baseline and KG-augmented variants.

The methodology produced a working VQA baseline, a bounded ConceptNet slicing pipeline, late-fusion mechanisms, full-validation evaluation, ablation results, a random-slice control, and reproducible result artefacts.

The final outcome was a negative but informative result: the implemented ConceptNet late-fusion branch did not improve OK-VQA validation accuracy over the frozen baseline. However, the methodology made this result defensible by showing how the system behaved under multiple fusion and control conditions.
