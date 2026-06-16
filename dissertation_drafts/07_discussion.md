# Discussion

## 1. Chapter Purpose

This chapter interprets the results of the project and explains what they show about bounded ConceptNet augmentation for OK-VQA. The project did not demonstrate an accuracy improvement from the implemented knowledge graph branch. However, the result is still informative because the system produced a controlled comparison between a frozen ViLT baseline and several knowledge-augmented late-fusion variants.

The purpose of this chapter is therefore not to re-present the results table. Instead, it explains why the observed pattern occurred, what can be learned from it, and how the project contributes as a system-development investigation even though the final quantitative result was negative.

## 2. Answer to the Research Question

The research question was:

> Does bounded, task-specific ConceptNet knowledge graph augmentation, integrated via late fusion, improve OK-VQA validation VQA-soft accuracy compared with a frozen ViLT baseline under matched conditions?

The answer for this implementation is no.

The final full-validation results did not show an improvement over the frozen baseline. The naive weighted fusion configuration substantially reduced performance. Top-N constrained weighted fusion reduced this harm to a near-zero negative delta, but still did not improve over the baseline. Gated fusion preserved baseline performance, but did not produce a positive gain. The random-slice control also did not improve over the baseline.

This means that the implemented KG branch was not useful enough, or not well aligned enough, to improve the frozen baseline's answer predictions. However, the evaluation does show useful behaviour across the different configurations. It shows that unconstrained KG fusion can be harmful, that constraining the answer space reduces harm, that gating can neutralise unreliable KG evidence, and that unrelated random KG slices do not create an artificial boost.

## 3. Interpretation of the Negative Result

A negative result in this project does not mean that the system failed to function. The system did what it was designed to do: it loaded OK-VQA data, evaluated a frozen baseline, built ConceptNet slices, encoded facts, fused knowledge-derived logits with baseline logits, ran ablations, and recorded matched baseline-versus-fused comparisons.

The negative result instead shows that the implemented form of ConceptNet augmentation was not sufficient to improve OK-VQA validation accuracy.

This distinction is important. The project was not simply an attempt to maximise benchmark performance. It was a controlled system-development investigation into whether a specific knowledge-augmentation design improved a fixed baseline. The evidence shows that this design did not improve the baseline, but the system and evaluation harness still provide a meaningful answer to the research question.

## 4. Why Naive Weighted Fusion Harmed Performance

The most important result is that naive weighted fusion substantially degraded performance. This suggests that directly adding KG-derived logits across the answer vocabulary can be dangerous when the KG signal is noisy, weakly relevant, or poorly calibrated.

The KG branch converts retrieved ConceptNet facts into an answer-vocabulary signal. For this to help, several things must all go right:

1. the question-derived entities must identify the concepts needed to answer the question;
2. ConceptNet must contain useful facts for those concepts;
3. the fact-ranking process must select relevant facts;
4. the fact encoder must represent those facts in a way that aligns with possible answers;
5. the fusion mechanism must add the KG signal at the right strength.

If any of these steps fail, the KG branch may push probability mass toward incorrect answers. The naive weighted result shows that this risk is real. The external knowledge branch was not neutral; when injected too broadly, it damaged the baseline prediction distribution.

This is an important finding because it challenges the simple assumption that adding commonsense knowledge will automatically improve VQA performance. External knowledge can help only if it is relevant, grounded, aligned, and calibrated.

## 5. Why Top-N Constrained Fusion Was Safer

Top-N constrained fusion reduced the harm caused by naive weighted fusion. Instead of allowing the KG branch to influence the full answer vocabulary, top-N fusion restricted its effect to the baseline model's most plausible candidate answers.

This design choice makes sense because the frozen baseline already contains visual and linguistic information from the image-question pair. If the KG branch is uncertain or noisy, it should not be allowed to promote arbitrary low-probability answers. Restricting fusion to the top-N baseline candidates limits the damage that KG noise can cause.

The results support this interpretation. Top-50 and top-20 constrained weighted fusion both avoided the large degradation seen in naive weighted fusion. However, they still produced a small negative delta. This means top-N constrained fusion made the system safer, but did not make the KG evidence useful enough to improve accuracy.

The implication is that answer-space constraint is helpful, but not sufficient. It controls where the KG signal can act, but it does not fix the underlying relevance or alignment problem.

## 6. Why Gated Fusion Preserved the Baseline

Gated fusion preserved baseline performance in the full-validation runs. This suggests that the gated configuration neutralised the KG contribution in aggregate when the KG evidence was not useful enough to improve predictions.

This should be interpreted carefully. Unless gate values are analysed directly, it should not be claimed that the model definitively learned a specific human-interpretable gating strategy. The safer interpretation is behavioural: the gated configuration did not allow the KG branch to damage the final accuracy.

This is still useful. It shows that gating can act as a protective mechanism when external knowledge is unreliable. In this project, gated fusion did not create a gain, but it avoided the large loss seen in naive weighted fusion.

The implication is that future KG-augmented VQA systems should treat external knowledge as optional evidence, not as automatically trustworthy evidence. A system should be able to suppress KG influence when retrieval is weak or ambiguous.

## 7. Interpretation of the Random-Slice Control

The random-slice control was included to test attribution. In this condition, the KG branch used unrelated but deterministic ConceptNet slices. If this condition had improved over the baseline, then any gain from the task-specific KG branch would be difficult to interpret because it might have come from extra parameters, reranking, or regularisation effects rather than useful knowledge.

The random-slice control did not improve over the frozen baseline. This supports the conclusion that the auxiliary KG branch did not create a spurious gain by itself.

In the final project, the task-specific KG branch also did not improve over the baseline. Therefore, the random-slice control does not prove that the task-specific retrieval was useful. Instead, it strengthens the validity of the evaluation by showing that unrelated KG content was not artificially boosting the model.

The random-slice control is one of the strongest parts of the project because it demonstrates awareness of attribution. It shows that the evaluation was not limited to asking whether accuracy changed, but also considered why an accuracy change might occur.

## 8. Likely Causes of the Result

The most likely explanation for the final result is a combination of retrieval, representation, and fusion limitations.

### 8.1 Entity Grounding Limitations

The KG branch relies primarily on entities extracted from the question text. This is a limitation for OK-VQA because the relevant concept may be visually present but not explicitly named in the question.

For example, a question may ask what activity is being performed, what object is used for a purpose, or what background knowledge is needed to interpret a scene. If the text does not contain the key visual object, question-only entity extraction may retrieve the wrong ConceptNet neighbourhood.

This means that KG slicing can be bounded and reproducible while still being semantically incomplete.

### 8.2 ConceptNet Relevance Limitations

ConceptNet contains broad commonsense relations, but not every retrieved fact is useful for answering a specific visual question. Many facts are generic, loosely related, or expressed at a level of abstraction that does not map cleanly to the expected short answer.

Relation filtering and top-k ranking reduce the amount of noise, but they do not guarantee that the final selected facts are the right facts. The results suggest that the retrieved ConceptNet evidence was often not strong enough to improve the answer distribution.

### 8.3 Answer-Space Alignment Limitations

The KG facts must ultimately influence a fixed answer vocabulary. A fact can be semantically relevant in natural language but still fail to align with the exact answer expected by the VQA scorer.

For example, a retrieved fact may imply an answer indirectly, use a synonym not present in the answer vocabulary, or describe a concept related to the answer without matching the answer string. The knowledge encoder may therefore produce a signal that is plausible but not useful for the final VQA-soft score.

### 8.4 Calibration Limitations

Late fusion requires the KG-derived signal to be calibrated relative to the baseline logits. If the KG signal is too strong, it can override useful baseline predictions. If it is too weak, it has no practical effect. If it is misaligned, even a small contribution can harm marginal cases.

The naive weighted result shows that calibration matters. The top-N and gated results show that safer fusion mechanisms can reduce damage, but they do not solve the upstream quality problem.

## 9. Contribution of the Project

The project's contribution is not a new state-of-the-art VQA score. Its contribution is a controlled system-development study of KG augmentation for OK-VQA.

The project contributes:

1. an end-to-end baseline-plus-KG VQA pipeline;
2. bounded ConceptNet slice construction;
3. configuration-aware slice caching;
4. weighted and gated late-fusion mechanisms;
5. top-N constrained fusion;
6. full-validation baseline-versus-fused comparison;
7. an ablation matrix over KG and fusion settings;
8. a random-slice attribution control;
9. final result packaging with traceable run IDs and metrics.

These contributions support a defensible conclusion even though the result is negative. The project shows how a KG-augmented VQA system can be built and evaluated in a way that exposes when knowledge helps, harms, or is neutralised.

## 10. Relationship to the Literature

The result is consistent with a broader theme in knowledge-augmented VQA: external knowledge is potentially useful, but only when retrieval, grounding, and fusion are effective.

OK-VQA requires knowledge beyond direct image recognition, so it is reasonable to explore external knowledge sources. However, this project shows that simply adding structured commonsense facts is not enough. The facts must be connected to the right image-question context and must influence the answer space in a controlled way.

The findings also support the design motivation for late fusion and controls. Because external knowledge can be noisy, a modular design makes it possible to evaluate the baseline and KG branch separately, compare fusion methods, and include controls such as random slices.

In the final dissertation, this chapter should connect directly back to the literature review. The literature should explain why knowledge is needed for OK-VQA, while this discussion explains why the implemented knowledge branch did not produce a gain.

## 11. Implications for Future Work

The results suggest several directions for future work.

### 11.1 Better Entity Grounding

Future work should include visual concepts, object detections, captions, or scene descriptions in the KG retrieval process. This could reduce the limitations of question-only entity extraction.

### 11.2 Improved Fact Ranking

The fact-ranking method could be improved using stronger semantic relevance models, question-answer compatibility scoring, or learned retrieval. This may help distinguish useful commonsense facts from generic ConceptNet neighbours.

### 11.3 Fact-to-Answer Alignment

Future work should improve the mapping between retrieved facts and answer candidates. This could include synonym expansion, answer normalisation improvements, entailment-style scoring, or direct ranking of candidate answers conditioned on facts.

### 11.4 Gate Analysis

The gated fusion results should be analysed more deeply in future work. Logging and visualising gate values could show when the model suppresses or allows KG evidence and whether this behaviour correlates with slice quality.

### 11.5 Manual Slice Quality Evaluation

A manual analysis of KG slices would strengthen the interpretation of the quantitative results. For a sample of questions, future work could label whether the retrieved facts are relevant, partially relevant, irrelevant, or misleading.

### 11.6 Stronger Baseline and Alternative Knowledge Sources

Future work could repeat the experiment with a stronger baseline or alternative knowledge sources. However, this should be done only after improving grounding and answer alignment; otherwise, additional knowledge may continue to act as noise.

## 12. Limitations of the Project

Several limitations should be acknowledged.

The baseline is a local project baseline rather than a state-of-the-art OK-VQA model. This limits claims about absolute benchmark competitiveness, but the matched comparison remains valid for testing the KG branch within this system.

The ablation matrix uses a smaller 512-example validation subset. It is useful for directional analysis but should not be treated as the main performance evidence.

The KG branch uses question-derived entities, which can miss important visual concepts.

The project evaluates whether KG fusion changes answer accuracy, but it does not include a full manual annotation study of slice quality.

The random-slice control tests for spurious gains from unrelated KG content, but it does not prove that the task-specific slices are semantically optimal.

These limitations do not invalidate the project. They define the boundary of the claims that can be made.

## 13. Overall Discussion Summary

The project provides a clear answer to the research question. The implemented bounded ConceptNet late-fusion branch did not improve OK-VQA validation VQA-soft accuracy over the frozen ViLT baseline.

However, the results are informative rather than empty. They show that naive KG injection can harm performance, that top-N constrained fusion can reduce this harm, that gated fusion can preserve baseline performance, and that random KG evidence does not create a spurious boost.

The main lesson is that knowledge augmentation is not automatically beneficial. For OK-VQA, external knowledge must be grounded in the image-question context, relevant to the reasoning need, aligned with the answer vocabulary, and calibrated against the baseline model. The current system achieved controlled integration and evaluation, but not the level of KG relevance and alignment required to improve accuracy.

This makes the project a defensible system-development contribution: it built the system, evaluated it under controlled conditions, and produced a clear, honest, and useful conclusion.
