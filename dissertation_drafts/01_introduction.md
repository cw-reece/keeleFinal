# Introduction

## 1.1 Background

Visual Question Answering (VQA) is the task of answering natural-language questions about images. A VQA system must combine visual understanding with language understanding in order to produce a short answer. Some questions can be answered directly from visible image content, such as identifying an object, colour, action, or count. Other questions require knowledge that is not fully visible in the image. These questions require the system to connect visual and textual evidence with external commonsense or world knowledge.

OK-VQA focuses on this harder form of visual question answering. It contains image-question pairs where answering often requires knowledge beyond object recognition. For example, a question may require understanding what an object is used for, where an activity usually takes place, or what commonsense relationship connects objects in the image. This makes OK-VQA suitable for investigating whether structured external knowledge can improve VQA performance.

Knowledge graphs are one possible way to provide external knowledge. ConceptNet is a commonsense knowledge graph containing relations between everyday concepts. In principle, it can provide useful information for VQA questions that require commonsense reasoning. However, using a knowledge graph is not automatically beneficial. Retrieved facts may be irrelevant, too generic, weakly connected to the question, or poorly aligned with the short answer expected by the VQA evaluation metric. Knowledge augmentation can therefore introduce noise as well as useful evidence.

This project investigates that tension. It asks whether bounded, task-specific ConceptNet knowledge graph augmentation, integrated through late fusion, improves OK-VQA validation performance compared with a frozen ViLT baseline under matched conditions.

## 1.2 Problem Statement

Modern vision-language models can answer many image-based questions, but they may struggle when the required answer depends on external knowledge rather than direct visual recognition. Knowledge graph augmentation is a plausible solution because it can provide structured commonsense facts. However, external knowledge can also harm performance if the retrieved facts are noisy, irrelevant, or badly aligned with the model's answer space.

The problem addressed by this project is therefore not simply how to add more knowledge to a VQA model. The more specific problem is how to build and evaluate a controlled knowledge-augmented VQA system where the effect of the knowledge branch can be measured against a stable baseline.

This creates several technical challenges:

- selecting a baseline model and treating it as a fixed comparator;
- extracting useful concepts from the question;
- retrieving a bounded and auditable ConceptNet slice;
- converting retrieved facts into a signal over the answer vocabulary;
- combining the baseline and knowledge signals without allowing noisy KG evidence to dominate;
- evaluating whether any change in performance is caused by task-specific knowledge rather than by extra parameters, reranking, or experimental noise.

The project addresses these challenges by implementing a modular OK-VQA pipeline with a frozen ViLT baseline, bounded ConceptNet slicing, late fusion, ablation settings, and a random-slice control.

## 1.3 Research Question

The fixed research question is:

> Does bounded, task-specific ConceptNet knowledge graph augmentation, integrated via late fusion, improve OK-VQA validation VQA-soft accuracy compared with a frozen ViLT baseline under matched conditions?

This question is deliberately specific. It does not ask whether all external knowledge improves VQA, whether ConceptNet is the best knowledge source, or whether the system reaches state-of-the-art performance. It asks whether this implemented bounded ConceptNet late-fusion approach improves a controlled local baseline.

The research question requires a comparative evaluation. The baseline must be frozen so that any change in performance can be attributed to the knowledge branch and fusion strategy rather than to changes in the underlying vision-language model.

## 1.4 Aim and Objectives

The aim of the project is to design, implement, and evaluate a reproducible knowledge-augmented VQA system that tests whether bounded ConceptNet knowledge improves OK-VQA validation accuracy over a frozen ViLT baseline.

The project objectives are:

1. Build an OK-VQA data pipeline that loads image-question-answer examples and supports validation evaluation.
2. Establish a ViLT-based baseline model over a fixed answer vocabulary.
3. Freeze a selected baseline checkpoint for matched comparison.
4. Implement ConceptNet-based entity lookup and bounded knowledge graph slice construction.
5. Cache slices using configuration-aware keys to support deterministic reruns and ablations.
6. Encode retrieved KG facts into an answer-vocabulary signal.
7. Implement late-fusion strategies, including weighted fusion, gated fusion, and top-N constrained fusion.
8. Evaluate full-validation baseline and fusion runs using VQA-style soft accuracy.
9. Run diagnostic ablations over fusion mode, relation set, and top-k settings.
10. Run a random-slice control to test whether unrelated KG evidence produces spurious gains.
11. Produce result summaries, error-analysis support, and reproducible documentation suitable for dissertation reporting.

## 1.5 Project Scope

The project is a system-development investigation. Its contribution is the design, implementation, and controlled evaluation of a KG-augmented VQA pipeline. It is not intended to produce a state-of-the-art OK-VQA model.

The project includes:

- OK-VQA dataset loading and validation evaluation;
- a ViLT-based baseline answer classifier;
- a fixed answer vocabulary;
- ConceptNet retrieval through a local store;
- question-derived entity extraction;
- bounded slice construction using hop depth, top-k, relation filters, and neighbour limits;
- fact encoding with sentence embeddings;
- weighted and gated late fusion;
- top-N constrained reranking;
- full-validation result comparison;
- ablation experiments;
- random-slice control;
- result logging and dissertation-ready summaries.

The project excludes:

- training a new large vision-language foundation model;
- claiming state-of-the-art OK-VQA performance;
- using large language model prompting as the main answer generator;
- using additional knowledge sources such as Wikidata;
- manual annotation of slice quality as a primary evaluation method;
- deployment of the model as a production VQA service.

These exclusions keep the project bounded and allow the research question to remain focused on the implemented ConceptNet late-fusion approach.

## 1.6 Overview of the Implemented System

The implemented system has two main branches.

The first branch is the frozen VQA baseline. It uses a ViLT-based classifier to process the image and question and produce answer logits over a fixed answer vocabulary. This branch provides the comparison point for all KG-augmented experiments.

The second branch is the knowledge branch. It extracts candidate entities from the question, retrieves ConceptNet neighbours, filters and ranks facts, and constructs a bounded task-specific KG slice. The selected facts are encoded into a knowledge-derived answer signal.

The two branches are combined using late fusion. Weighted fusion directly adds a scaled KG signal to the baseline logits. Gated fusion allows the model to suppress unreliable KG evidence. Top-N constrained fusion restricts KG influence to the baseline model's most plausible answer candidates.

The system is evaluated using OK-VQA validation VQA-soft accuracy. The final evaluation includes full-validation runs, a 512-example diagnostic ablation matrix, and a full-validation random-slice control.

## 1.7 Summary of Final Findings

The final evaluation does not show an improvement from bounded task-specific ConceptNet augmentation via late fusion.

The frozen ViLT baseline achieved a validation VQA-soft accuracy of 0.163892 on 5046 OK-VQA validation examples. Naive weighted KG fusion reduced performance substantially. Top-N constrained weighted fusion reduced the harm to a near-zero negative delta but still did not improve over the baseline. Gated fusion preserved baseline performance, suggesting that the gated configuration neutralised unreliable KG influence in aggregate. The random-slice control did not improve over the baseline, indicating that unrelated KG evidence did not create an artificial gain.

The strongest conclusion is therefore diagnostic rather than performance-improving. The implemented system shows that bounded ConceptNet slicing and late fusion can be built and evaluated reproducibly, but the current KG retrieval and answer-alignment approach does not improve OK-VQA validation accuracy over the frozen baseline. The results suggest that future work should focus on stronger entity grounding, better slice relevance, improved fact-answer alignment, and more effective calibration of the KG signal.

## 1.8 Contribution

The project makes the following contributions:

1. A reproducible system-development implementation of a knowledge-augmented OK-VQA pipeline.
2. A frozen ViLT baseline used for matched comparison.
3. A bounded ConceptNet slice builder with configurable hop depth, top-k selection, relation filtering, neighbour limits, and cache hashing.
4. A KG fact encoder that maps selected facts into answer-vocabulary logits.
5. Weighted, gated, and top-N constrained late-fusion mechanisms.
6. Full-validation comparison between frozen baseline and KG-augmented variants.
7. A diagnostic ablation matrix over KG and fusion settings.
8. A random-slice control for testing whether unrelated KG evidence produces spurious gains.
9. A clear negative result showing that the implemented KG branch did not improve validation accuracy, together with an explanation of likely failure modes.

The main contribution is not an accuracy improvement. The main contribution is a controlled, auditable system and evaluation showing how bounded ConceptNet late fusion behaves under matched OK-VQA conditions.

## 1.9 Dissertation Structure

The remainder of the dissertation is organised as follows.

Chapter 2 reviews literature on Visual Question Answering, OK-VQA, external knowledge augmentation, ConceptNet, and fusion strategies.

Chapter 3 explains the methodology used to develop and evaluate the system, including the baseline-first and experiment-driven approach.

Chapter 4 presents the requirements and design of the system, including functional requirements, non-functional requirements, architecture, and traceability.

Chapter 5 describes the implementation of the OK-VQA pipeline, baseline model, ConceptNet slicing module, knowledge encoder, fusion mechanisms, and evaluation scripts.

Chapter 6 presents the testing and evaluation, including full-validation results, ablations, random-slice control, error analysis, and threats to validity.

Chapter 7 discusses the findings, limitations, interpretation of the negative result, and implications for knowledge-augmented VQA.

Chapter 8 concludes the dissertation and outlines future work.
