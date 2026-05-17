# Final Results Summary

## Purpose

This file consolidates the main quantitative evidence used for the dissertation results and evaluation chapter. It separates full-validation runs from smaller ablation-matrix runs so that the dissertation does not overstate the strength of subset experiments.

## Full-validation baseline and main fusion runs

| Run | Setting | Fusion mode | Validation examples | Baseline acc | Fused acc | Delta |
|---|---|---|---:|---:|---:|---:|
| BASELINE_FREEZE_20260312_1456 | Frozen ViLT baseline | none | 5046 | 0.163892 | n/a | n/a |
| 20260313_140303_m5_fusion_weighted_fullval | Naive full-space KG fusion | weighted | 5046 | 0.163892 | 0.110517 | -0.053376 |
| 20260313_141000_m5_gated_fullval | Naive gated KG fusion | gated | 5046 | 0.163892 | 0.163892 | 0.000000 |
| 20260313_143730_m5_topn_weighted_fullval | Top-50 constrained fusion | weighted | 5046 | 0.163892 | 0.163760 | -0.000132 |
| 20260313_145031_m5_topn20_weighted_fullval | Top-20 constrained fusion | weighted | 5046 | 0.163892 | 0.163760 | -0.000132 |
| 20260313_145705_m5_topn20_gated_fullval | Top-20 gated fusion | gated | 5046 | 0.163892 | 0.163892 | 0.000000 |

## Full-validation interpretation

The naive weighted fusion run substantially reduced performance, falling from 0.163892 to 0.110517. This indicates that unconstrained KG fusion can inject harmful noise into the answer distribution.

The gated fusion run preserved baseline performance exactly in the full-validation setting, suggesting that the gating mechanism learned to suppress unreliable KG evidence rather than improve the baseline.

Top-N constrained weighted fusion avoided the large degradation seen in naive weighted fusion, but it still did not improve over the frozen baseline. Both top-50 and top-20 constrained weighted runs produced a small negative delta of -0.000132.

Overall, the full-validation results do not support a claim that the implemented ConceptNet late-fusion branch improves OK-VQA validation accuracy over the frozen ViLT baseline. They do support a more diagnostic conclusion: bounded and gated fusion can reduce the harm caused by noisy KG evidence, but the current KG retrieval and answer-alignment method does not produce a reliable positive gain.

## Ablation matrix on 512-example validation subset

These runs use a smaller validation subset of 512 examples. They are useful for comparing settings directionally, but they should not be reported as final headline performance.

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

## Ablation interpretation

The ablation matrix shows no positive accuracy gain from the KG branch across the tested relation sets, top-K values, and fusion modes.

Weighted fusion is consistently slightly negative on the 512-example subset. The negative deltas are small, but they show that adding KG logits tends to perturb the baseline in the wrong direction.

Gated fusion is safer than weighted fusion. In the strict top-K 20 and broad top-K 20 settings, gated fusion preserved the baseline exactly. This supports the interpretation that gating can act as a safety mechanism when KG evidence is not reliable enough to improve prediction.

The strict and broad relation settings do not show a meaningful difference in this subset. Both produce similar near-zero or slightly negative deltas. This suggests that relation-set choice alone was not sufficient to make the ConceptNet evidence useful under the current entity extraction and knowledge-encoding approach.

## Random-slice control

| Condition | VQA-soft accuracy |
|---|---:|
| Frozen ViLT baseline | 0.163892 |
| Random-slice fused | 0.163760 |
| Delta | -0.000132 |

## Random-slice interpretation

The random-slice control did not improve over the frozen baseline. This suggests that the presence of an auxiliary KG branch and fusion mechanism alone was not sufficient to create a performance gain.

This supports the attribution argument: if future task-specific KG runs show improvement, the improvement should not be assumed to come merely from extra trainable fusion parameters, auxiliary logits, or reranking side effects. In the current project, the random-slice result also reinforces the broader finding that the implemented KG branch does not create an artificial performance boost.

## Overall answer to the research question

The current implementation does not demonstrate an improvement in OK-VQA validation VQA-soft accuracy from bounded task-specific ConceptNet augmentation via late fusion.

The strongest defensible conclusion is:

Bounded ConceptNet slicing and late fusion were successfully implemented and evaluated under matched conditions, but the KG branch did not improve validation accuracy over the frozen ViLT baseline. Naive weighted fusion substantially degraded performance, constrained weighted fusion reduced that degradation to a near-zero negative delta, and gated fusion preserved baseline performance by suppressing unreliable KG evidence. The random-slice control confirmed that unrelated KG evidence did not create a spurious improvement. These findings suggest that the main bottleneck is not the late-fusion mechanism itself, but the relevance and answer-space alignment of the retrieved ConceptNet evidence.

## Dissertation reporting notes

Use the full-validation results for the headline conclusion.

Use the 512-example ablation matrix as supporting diagnostic evidence only.

Do not claim that ConceptNet improves OK-VQA accuracy.

Do claim that the system provides a reproducible evaluation harness and diagnostic evidence showing when KG augmentation harms, is suppressed, or has no measurable effect.
