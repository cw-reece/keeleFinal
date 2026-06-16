# Conclusion

## 1. Chapter Purpose

This chapter summarises the project, answers the research question, reflects on the main findings, and identifies future work. The project investigated whether bounded, task-specific ConceptNet knowledge graph augmentation, integrated through late fusion, improves OK-VQA validation performance compared with a frozen ViLT baseline.

The project should be understood as a system-development investigation rather than a state-of-the-art benchmark submission. Its contribution is the design, implementation, and evaluation of a reproducible KG-augmented VQA pipeline with matched baseline comparison, ablation experiments, and a random-slice control.

## 2. Research Question

The research question was:

> Does bounded, task-specific ConceptNet knowledge graph augmentation, integrated via late fusion, improve OK-VQA validation VQA-soft accuracy compared with a frozen ViLT baseline under matched conditions?

The answer for the implemented system is no. The ConceptNet late-fusion branch did not improve OK-VQA validation VQA-soft accuracy over the frozen ViLT baseline.

However, the result is not simply a failed improvement attempt. The system produced useful diagnostic evidence about how external KG evidence behaves when added to a frozen VQA baseline. The evaluation showed that naive weighted fusion can substantially harm performance, top-N constrained fusion can reduce that harm, gated fusion can preserve baseline performance, and a random-slice control does not create a spurious improvement.

## 3. Summary of Work Completed

The project completed the main system-development objectives.

First, an OK-VQA data pipeline was implemented to load question, annotation, image, and answer data for training and evaluation.

Second, a ViLT-based baseline model was trained and frozen for matched comparison. The final frozen baseline used for dissertation reporting was `BASELINE_FREEZE_20260312_1456`, with validation VQA-soft accuracy of 0.163892 on 5046 OK-VQA validation examples.

Third, a ConceptNet knowledge branch was implemented. This included question entity extraction, ConceptNet neighbour retrieval, relation filtering, bounded slice construction, fact scoring, top-k selection, and caching using configuration-aware keys.

Fourth, a knowledge encoder was implemented to convert retrieved ConceptNet facts into a KG-derived answer signal over the fixed answer vocabulary.

Fifth, late-fusion mechanisms were implemented, including weighted fusion, gated fusion, and top-N constrained reranking.

Sixth, the project implemented an evaluation harness that compared baseline and fused predictions under matched conditions. It logged baseline accuracy, fused accuracy, delta, configuration, validation size, fusion mode, and slice settings.

Seventh, the project added ablation experiments and a random-slice control to support diagnostic interpretation and avoid overclaiming.

## 4. Main Findings

The main finding is that the implemented KG branch did not improve the frozen baseline.

The full-validation baseline achieved VQA-soft accuracy of 0.163892. Naive weighted KG fusion reduced accuracy to 0.110517, a delta of -0.053376. This indicates that unconstrained KG injection can seriously harm performance when the external knowledge signal is noisy, poorly calibrated, or poorly aligned with the answer vocabulary.

Top-N constrained weighted fusion reduced the damage substantially. Both top-50 and top-20 constrained weighted fusion produced a much smaller negative delta of -0.000132. This suggests that restricting KG influence to plausible baseline answers is safer than allowing the KG branch to perturb the full answer distribution.

Gated fusion preserved baseline accuracy in the full-validation runs. This suggests that the gated configuration neutralised the harmful KG contribution in aggregate. Without detailed gate-value analysis, this should be treated as behavioural evidence rather than direct proof of the internal gate mechanism.

The random-slice control did not improve over the frozen baseline. This supports the interpretation that the presence of an auxiliary KG branch, additional trainable parameters, or top-N reranking did not create an artificial gain.

Overall, the results suggest that the main bottleneck was not simply the late-fusion mechanism. The more important limitation was the relevance, grounding, and answer-space alignment of the retrieved ConceptNet evidence.

## 5. Contribution

The project makes four main contributions.

The first contribution is a modular KG-augmented VQA system that separates the baseline VQA branch from the external knowledge branch. This separation allowed controlled comparison between baseline-only and KG-augmented predictions.

The second contribution is a bounded ConceptNet slicing pipeline. Rather than retrieving unrestricted graph neighbourhoods, the system uses configurable controls such as hop depth, relation set, top-k fact count, neighbour limits, and cache keys. This makes the KG retrieval process auditable and reproducible.

The third contribution is a controlled evaluation framework. The project compares baseline and fused predictions under matched conditions and records results in configuration-linked run artefacts.

The fourth contribution is diagnostic evidence about KG fusion behaviour. The results show that naive KG fusion can harm performance, constrained fusion can reduce harm, gated fusion can preserve baseline performance, and unrelated random KG slices do not create spurious improvement.

## 6. Limitations

The project has several limitations.

The baseline is a local project baseline rather than a state-of-the-art OK-VQA system. This limits the absolute benchmark significance of the accuracy values, but it does not invalidate the controlled comparison between the frozen baseline and the KG-augmented variants.

The KG branch relies heavily on question-derived entity extraction. If important visual concepts are present in the image but not explicitly named in the question, the ConceptNet slice may be built around incomplete or misleading concepts.

ConceptNet facts can be generic, noisy, or only weakly related to the required OK-VQA answer. Bounded retrieval improves auditability, but it does not guarantee semantic relevance.

The knowledge encoder maps retrieved facts into the answer vocabulary through semantic similarity. This means that even relevant facts may fail to improve predictions if they do not align well with the short answer strings expected by the VQA scoring process.

The ablation matrix used a smaller 512-example validation subset. These results are useful for diagnosis, but the full-validation runs should be treated as the headline evidence.

The random-slice control tests whether unrelated KG evidence creates spurious improvement, but it does not directly prove that task-specific slices are semantically high quality. A manual slice-quality annotation study would strengthen the analysis.

## 7. Future Work

Future work should focus on improving the quality and alignment of the KG evidence before expecting consistent accuracy gains.

One direction would be to improve entity grounding. The current approach relies mainly on question text. Adding object detections, image captions, or visual concept extraction could help the KG branch retrieve knowledge about visually important entities that are not named in the question.

A second direction would be to improve fact ranking. The current retrieval method uses bounded ConceptNet neighbourhoods and scoring, but stronger semantic ranking could help prioritise facts that are more likely to support the required answer.

A third direction would be to improve answer-space alignment. Retrieved facts may be relevant in natural language but still fail to support the exact short answer expected by OK-VQA. Future work could explore answer-aware fact selection or learned mappings between KG facts and likely VQA answers.

A fourth direction would be to evaluate slice quality directly. Manual annotation of a sample of KG slices could distinguish between retrieval failure, encoding failure, and fusion failure.

A fifth direction would be to compare ConceptNet with other knowledge sources or retrieval methods, such as caption-based retrieval, curated visual commonsense resources, or language-model-generated candidate facts. These approaches may provide more context-specific evidence than ConceptNet neighbourhood expansion alone.

## 8. Final Reflection

The project began with the hypothesis that bounded ConceptNet augmentation might improve OK-VQA performance when integrated through late fusion. The final results did not support that hypothesis for this implementation.

The important outcome is that the project produced a controlled and reproducible answer. The KG branch was implemented, evaluated, ablated, and tested with a random-slice control. The negative result is therefore informative: it shows that adding external commonsense knowledge is not automatically beneficial, and that KG relevance, grounding, calibration, and answer-space alignment are central challenges.

The final conclusion is that bounded ConceptNet slicing and late fusion can be implemented in a modular and auditable VQA pipeline, but the current implementation does not improve OK-VQA validation accuracy over the frozen ViLT baseline. The project therefore provides a useful system-development case study in both the promise and the difficulty of knowledge-augmented VQA.
