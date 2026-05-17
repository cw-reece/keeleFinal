# Testing and Evaluation

## 1. Evaluation Aim

The purpose of the evaluation was to determine whether bounded, task-specific ConceptNet knowledge graph augmentation, integrated through late fusion, improves OK-VQA validation performance compared with a frozen ViLT baseline under matched experimental conditions.

The evaluation was designed to test both system performance and attribution. A simple comparison between a baseline model and a knowledge-augmented model would not be sufficient, because any difference could be caused by additional trainable parameters, reranking effects, or noisy regularisation rather than useful external knowledge. For that reason, the evaluation includes a frozen baseline, full-validation fusion comparisons, smaller ablation runs, and a random-slice control.

The primary metric was VQA-style soft accuracy on the OK-VQA validation split. This metric reflects the standard VQA scoring approach in which a prediction receives partial credit depending on how many human annotators gave the same answer.

## 2. Evaluation Setup

The baseline system used a ViLT-based answer classifier over a fixed answer vocabulary. The baseline was treated as frozen for comparison against knowledge-augmented variants. This ensured that changes in performance could be attributed to the knowledge branch and fusion strategy rather than changes in the underlying vision-language model.

The knowledge branch used ConceptNet as the external knowledge source. For each image-question pair, the system extracted candidate entities from the question, retrieved ConceptNet neighbours, applied relation filtering, ranked facts, and selected a bounded top-k slice. The selected facts were encoded into a knowledge-derived answer signal and combined with the baseline answer logits using late fusion.

The main fusion settings evaluated were:

- naive weighted fusion over the full answer space;
- gated fusion over the full answer space;
- top-N constrained weighted fusion;
- top-N constrained gated fusion.

A smaller ablation matrix varied relation set, top-k fact count, and fusion mode. A random-slice control was also run to test whether unrelated KG evidence could produce a spurious improvement.

## 3. Testing Strategy

Testing was performed at three levels: component testing, integration testing, and system-level evaluation.

At the component level, the project required correct behaviour for answer normalisation, VQA-soft scoring, entity extraction, ConceptNet relation filtering, slice construction, cache-key generation, and deterministic random-slice generation. These checks were important because errors in answer normalisation or cache reuse could make reported accuracy misleading.

At the integration level, the key concern was whether data and predictions flowed correctly through the complete pipeline. The OK-VQA dataset loader had to produce image-question-answer records in the expected format. The baseline model had to output answer logits over the same answer vocabulary used by the evaluator. The KG branch had to build slices for the same question and image identifiers being evaluated. Finally, the fusion module had to combine baseline and KG logits without changing the baseline configuration.

At the system level, the evaluation tested complete saved runs. Each run logged its configuration, fusion mode, slice settings, random seed, validation size, baseline accuracy, fused accuracy, and delta. This allowed comparisons to be traced back to specific run IDs rather than informal experiment notes.

## 4. Full-Validation Results

The full-validation experiments used 5046 OK-VQA validation examples. These runs provide the headline results for answering the research question.

| Run | Setting | Fusion mode | Validation examples | Baseline acc | Fused acc | Delta |
|---|---|---|---:|---:|---:|---:|
| BASELINE_FREEZE_20260312_1456 | Frozen ViLT baseline | none | 5046 | 0.163892 | n/a | n/a |
| 20260313_140303_m5_fusion_weighted_fullval | Naive full-space KG fusion | weighted | 5046 | 0.163892 | 0.110517 | -0.053376 |
| 20260313_141000_m5_gated_fullval | Naive gated KG fusion | gated | 5046 | 0.163892 | 0.163892 | 0.000000 |
| 20260313_143730_m5_topn_weighted_fullval | Top-50 constrained fusion | weighted | 5046 | 0.163892 | 0.163760 | -0.000132 |
| 20260313_145031_m5_topn20_weighted_fullval | Top-20 constrained fusion | weighted | 5046 | 0.163892 | 0.163760 | -0.000132 |
| 20260313_145705_m5_topn20_gated_fullval | Top-20 gated fusion | gated | 5046 | 0.163892 | 0.163892 | 0.000000 |

The naive weighted fusion run substantially degraded performance, reducing VQA-soft accuracy from 0.163892 to 0.110517. This indicates that directly injecting the KG-derived answer signal into the full answer distribution can be harmful. The result is consistent with the expectation that external KG evidence may introduce noise when entity linking, fact relevance, or answer-space alignment are imperfect.

The gated fusion run preserved baseline performance exactly. This suggests that the gating mechanism learned to suppress the KG branch when the external evidence was not useful enough to improve predictions. While this did not produce a positive gain, it is still an important diagnostic result because it shows that gating can reduce the risk of harmful KG injection.

Top-N constrained weighted fusion avoided the large degradation observed in naive weighted fusion. Both top-50 and top-20 constrained weighted runs produced a much smaller negative delta of -0.000132. This suggests that restricting KG influence to the baseline model’s most plausible answer candidates makes fusion safer, but still not beneficial under the current knowledge representation.

The full-validation results therefore do not support the hypothesis that the implemented bounded ConceptNet late-fusion branch improves OK-VQA validation accuracy over the frozen ViLT baseline.

## 5. Ablation Matrix

A smaller ablation matrix was run on a 512-example validation subset. These runs are not used as headline performance claims, but they help identify whether relation filtering, top-k size, or fusion mode changed the direction of the result.

| Fusion mode | Relation set | Top-K | Validation examples | Baseline acc | Fused acc | Delta |
|---|---|---:|---:|---:|---:|---:|
| weighted | strict | 5 | 512 | 0.135417 | 0.134115 | -0.001302 |
| weighted | strict | 10 | 512 | 0.135417 | 0.134766 | -0.000651 |
| weighted | strict | 20 | 512 | 0.135417 | 0.134115 | -0.001302 |
| gated | strict | 5 | 512 | 0.135417 | 0.134766 | -0.000651 |
| gated | strict | 10 | 512 | 0.135417 | 0.134766 | -0.000651 |
| gated | strict | 20 | 512 | 0.135417 | 0.135417 | 0.000000 |
| weighted | broad | 5 | 512 | 0.135417 | 0.134115 | -0.001302 |
| weighted | broad | 10 | 512 | 0.135417 | 0.134766 | -0.000651 |
| weighted | broad | 20 | 512 | 0.135417 | 0.134115 | -0.001302 |
| gated | broad | 5 | 512 | 0.135417 | 0.134766 | -0.000651 |
| gated | broad | 10 | 512 | 0.135417 | 0.134766 | -0.000651 |
| gated | broad | 20 | 512 | 0.135417 | 0.135417 | 0.000000 |

The ablation matrix shows no positive accuracy gain across the tested settings. Weighted fusion was consistently slightly negative. Gated fusion was safer, preserving baseline performance in the top-k 20 settings for both strict and broad relation sets.

The strict and broad relation settings did not produce a meaningful difference in this subset. This suggests that simply widening or narrowing the ConceptNet relation set was not enough to make the KG branch useful. The likely bottleneck is deeper: the retrieved facts may not align well with the required answer vocabulary, or the question-derived entity extraction may not reliably select the visual or commonsense concepts needed for OK-VQA reasoning.

## 6. Random-Slice Control

A random-slice control was included to test attribution. In this condition, ConceptNet slices were generated deterministically but independently of the question content. The purpose was to check whether the auxiliary KG branch and fusion mechanism could improve accuracy even when the retrieved knowledge was unrelated to the question.

| Condition | VQA-soft accuracy |
|---|---:|
| Frozen ViLT baseline | 0.163892 |
| Random-slice fused | 0.163760 |
| Delta | -0.000132 |

The random-slice control did not improve over the frozen baseline. This supports the conclusion that the presence of an additional KG branch, extra trainable fusion parameters, and top-N reranking did not create an artificial performance gain.

This control strengthens the interpretation of the main results. If a task-specific KG run had improved accuracy, the random-slice result would help argue that the improvement was more likely due to relevant knowledge retrieval. In the current project, the control instead reinforces the broader finding that the implemented KG branch does not generate a spurious improvement.

## 7. Error Analysis and Diagnostic Interpretation

The quantitative results show that the implemented KG branch did not improve validation accuracy. The likely causes are technical rather than conceptual.

First, ConceptNet contains many generic commonsense relations. Even when relation filtering and top-k bounds are used, retrieved facts can be too broad or only weakly related to the answer required by OK-VQA.

Second, entity extraction from question text alone can miss visually important entities. If the question asks about an object or scene property that is not explicitly named, the KG slice may be built around incomplete or misleading seed concepts.

Third, KG facts do not necessarily align with the answer vocabulary. A retrieved fact may be semantically relevant but still fail to push probability mass toward the exact short answer expected by the VQA scorer.

Fourth, late fusion can only help if the KG-derived signal is both relevant and calibrated. The naive weighted fusion result shows that an uncalibrated KG signal can strongly damage performance. The gated and top-N constrained results show safer behaviour, but safety alone does not create accuracy improvement.

## 8. Threats to Validity

Several threats to validity should be noted.

The ablation matrix used a 512-example validation subset, so those results should be treated as diagnostic rather than definitive. The full-validation runs provide the headline conclusion.

The baseline accuracy is below modern state-of-the-art OK-VQA performance. However, the project’s research question is comparative within a controlled implementation: whether the bounded KG branch improves over the frozen baseline under matched conditions.

The random-slice control tests whether unrelated knowledge creates a spurious gain, but it does not prove that task-specific slices are semantically correct. Additional manual slice-quality annotation would strengthen the analysis.

The evaluation uses VQA-soft accuracy, which is appropriate for OK-VQA but still depends on answer normalisation and exact answer vocabulary coverage. Some semantically reasonable answers may receive low credit if they do not match the annotator answer distribution.

## 9. Evaluation Conclusion

The evaluation does not demonstrate an accuracy improvement from bounded task-specific ConceptNet augmentation via late fusion.

The strongest defensible conclusion is that the system successfully implemented and evaluated a modular KG-augmented VQA architecture, but the current knowledge retrieval and fusion approach did not improve OK-VQA validation accuracy. Naive weighted fusion harmed performance, top-N constrained fusion reduced the harm to a near-zero negative delta, gated fusion preserved baseline performance by suppressing unreliable KG evidence, and the random-slice control confirmed that unrelated KG evidence did not create an artificial gain.

These findings suggest that future work should focus on improving entity grounding, KG slice relevance, fact-answer alignment, and calibration of the knowledge signal before expecting consistent gains from late-fusion ConceptNet augmentation.
