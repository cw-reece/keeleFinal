# Random Slice Control

## Purpose

This control tests whether the fusion system improves merely because an additional KG branch is present, rather than because retrieved ConceptNet evidence is relevant to the question.

The random-slice condition uses unrelated ConceptNet slices generated deterministically per question/image pair.

## Run configuration

- Fusion mode: weighted
- Top-N rerank: 20
- KG random slice: true
- Hop depth: 1
- Top-K facts: 10
- Relation set: strict
- Validation examples: 5046
- Seed: 42
- Epochs: 1
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Temperature: 2.0
- Runtime: 976.52 seconds

## Result

| Condition | VQA-soft accuracy |
|---|---:|
| Frozen ViLT baseline | 0.163892 |
| Random-slice fused | 0.163760 |
| Delta | -0.000132 |

## Interpretation

The random-slice fused model did not improve over the frozen baseline. The small negative delta indicates that unrelated KG evidence does not provide a meaningful benefit under the weighted top-20 fusion setting.

This strengthens the interpretation of the main KG experiments because any observed improvement from task-specific ConceptNet slices is less likely to be caused simply by adding an auxiliary KG branch, extra trainable fusion parameters, or reranking effects. Conversely, if the main KG experiments also show no improvement, this control helps show that the system is not being artificially boosted by irrelevant knowledge.