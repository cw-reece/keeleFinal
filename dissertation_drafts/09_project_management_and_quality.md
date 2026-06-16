# Project Management and Quality of Management

## 1. Purpose

This section records how the project was planned, managed, adapted, and quality checked. It supports the Quality of Management element of the CSC40098 system-development assessment by showing that the project was not only implemented, but also controlled, reviewed, and adjusted in response to evidence.

The project developed a knowledge-augmented Visual Question Answering system for OK-VQA. The original aim was to test whether bounded ConceptNet knowledge graph augmentation, integrated through late fusion, could improve validation VQA-soft accuracy compared with a frozen ViLT baseline. The final evaluation showed that the implemented KG branch did not improve accuracy, but the project still produced a complete system, controlled evaluation, ablation evidence, a random-slice control, and a defensible analysis of why KG augmentation was not beneficial in this implementation.

## 2. Project Management Approach

The project followed a baseline-first, experiment-driven development approach. This was chosen because the research question required controlled comparison. A knowledge-augmented model could only be evaluated meaningfully after a stable baseline had been established.

The work was divided into the following stages:

1. Project specification and literature grounding.
2. Baseline VQA system implementation.
3. ConceptNet knowledge slicing implementation.
4. Knowledge encoding and late-fusion integration.
5. Evaluation, ablation, and random-control experiments.
6. Result consolidation and dissertation packaging.
7. Demonstration planning and final submission preparation.

Each stage produced a concrete artefact such as source code, configuration files, metrics, result summaries, draft dissertation sections, or demonstration notes.

## 3. Milestone Summary

| Milestone | Planned purpose | Completed artefacts | Outcome |
|---|---|---|---|
| Project specification | Define the research question and system scope | Proposal, README, final checklist issue | Research question fixed around bounded ConceptNet late fusion |
| Literature review | Ground design in VQA, OK-VQA, KG augmentation, and fusion literature | Assessment 1 technical review, final literature review draft | Literature shaped bounded slicing, late fusion, and control design |
| Baseline implementation | Establish a stable comparator | ViLT baseline pipeline, answer vocabulary, baseline metrics | Frozen baseline selected for matched comparison |
| KG slicing | Retrieve bounded ConceptNet evidence per question | ConceptNet store, entity extraction, slice builder, caching | Inspectable KG slices produced |
| Fusion integration | Combine baseline and KG answer signals | Weighted fusion, gated fusion, top-N constrained fusion | KG branch could be enabled, disabled, and ablated |
| Evaluation | Test whether KG improves the frozen baseline | Full-validation runs, ablation matrix, random-slice control | KG did not improve accuracy; evidence supports a diagnostic conclusion |
| Result consolidation | Prevent stale or contradictory reporting | Final results summary, random-slice report, README cleanup | Final claim aligned across repo artefacts |
| Dissertation drafting | Convert system and evidence into report chapters | Draft chapters 1–8, appendices, demo notes | Main dissertation structure established |
| Demonstration preparation | Show system, evidence, and conclusion clearly | Demo script and defence notes | Demo can be recorded from stable artefacts |

## 4. Risk Register Summary

| Risk | Impact | Mitigation | Final status |
|---|---|---|---|
| Baseline instability | Fusion results would be hard to interpret | Freeze a selected baseline for matched comparison | Mitigated |
| Low baseline accuracy | Absolute benchmark performance may look weak | Frame project as controlled system-development comparison, not SOTA benchmarking | Accepted and explained |
| No KG improvement | Project could appear unsuccessful if overclaimed | Reframe as diagnostic evidence from a controlled system | Mitigated |
| Noisy ConceptNet retrieval | KG facts may harm predictions | Use bounded slices, relation filtering, top-k, top-N fusion, and gated fusion | Mitigated but remains a limitation |
| Cache contamination | Ablation results could reuse incompatible slices | Include slice-affecting configuration values in cache hash | Mitigated |
| Weak attribution | Improvements could be caused by extra branch/parameters rather than useful KG | Add random-slice control | Mitigated |
| Stale documentation | README or draft chapters could report old results | Run grep checks and align README/docs/reports | Mitigated |
| Overclaiming | Marker may challenge claims not supported by results | State clearly that KG did not improve accuracy | Mitigated |
| Demonstration failure | Live training or long commands could fail during recording | Use precomputed metrics and short demonstration commands | Mitigated |
| Time pressure | Final assembly could become rushed | Draft chapters early and keep final period for revision and packaging | Ongoing |

## 5. Change Control

Several design and reporting decisions changed during the project in response to evidence.

### Change 1: From improvement claim to diagnostic evaluation

The original motivating hypothesis was that bounded ConceptNet augmentation might improve OK-VQA accuracy. The final full-validation results did not support this. The dissertation was therefore adjusted to report a negative but informative result rather than forcing an unsupported improvement claim.

### Change 2: Addition of top-N constrained fusion

Naive weighted fusion substantially degraded accuracy. Top-N constrained fusion was introduced to reduce the risk of KG evidence perturbing the full answer distribution. This reduced the harm but did not create a positive gain.

### Change 3: Addition of gated fusion

Gated fusion was included to test whether the system could learn to suppress unreliable KG evidence. The final results showed that gated fusion preserved baseline performance, which supported the interpretation that safer fusion can reduce harm even when KG evidence is not useful enough to improve accuracy.

### Change 4: Addition of random-slice control

A random-slice control was added to test attribution. This ensured that the project could distinguish between useful task-specific KG evidence and possible side effects from adding an auxiliary branch or reranking mechanism.

### Change 5: Documentation alignment

Earlier baseline documentation reported an older development baseline. The README and baseline documentation were later aligned with the final frozen baseline and final results summary to avoid contradictory evidence.

## 6. Quality Assurance Activities

Quality assurance was performed through a mixture of code-level checks, experiment logging, repository review, and documentation consistency checks.

The main QA activities were:

- dataset integrity checks before training and evaluation;
- answer vocabulary construction and coverage checking;
- use of fixed run IDs and metrics files;
- frozen baseline comparison for fusion runs;
- logging of validation size, accuracy, delta, fusion mode, and KG configuration;
- random-slice control to test attribution;
- result consolidation under `reports/final_results/`;
- grep-based checks for stale baseline values and TODO markers;
- README cleanup so the repository landing page matched the final evidence;
- dissertation chapter drafting from final result artefacts rather than from memory.

## 7. Evidence Traceability

The project was managed so that major claims can be traced to artefacts.

| Claim | Supporting artefact |
|---|---|
| A frozen baseline was used for matched comparison | `BASELINE_FREEZE_20260312_1456` metrics |
| Naive weighted fusion harmed performance | `reports/final_results/final_results_summary.md` |
| Top-N constrained fusion reduced harm but did not improve accuracy | Full-validation top-20 and top-50 run summaries |
| Gated fusion preserved baseline performance | Full-validation gated run metrics |
| Random KG evidence did not create a spurious gain | `reports/final_results/random_slice_control.md` |
| The KG branch used bounded slicing | `src/kg/slice_builder.py` and KG configuration files |
| The knowledge encoder produced answer-vocabulary logits | `src/kg/knowledge_encoder.py` |
| The system supported reproducible evaluation | Run folders, config files, metrics JSON, result summaries |
| The final dissertation does not claim KG improvement | Final results summary, testing chapter, discussion, conclusion |

## 8. Reflection on Management Effectiveness

The most effective management decision was to freeze the baseline and keep the KG branch modular. This allowed the project to produce a clear answer even though the KG branch did not improve accuracy.

The second most effective decision was to add the random-slice control. Without it, the evaluation would have been weaker because it would be harder to explain whether any KG effect was caused by relevant knowledge or merely by the architecture.

The main weakness in management was that some documentation initially became stale as experiments changed. This was addressed through later consistency checks and README cleanup, but it shows the importance of maintaining a single source of truth for final results.

## 9. Final Management Position

At the final reporting stage, the project is managed as a completed system-development investigation with a controlled negative result. The evidence supports the conclusion that the implemented ConceptNet late-fusion branch did not improve OK-VQA validation accuracy, but the system and evaluation process were sufficiently complete to identify how and why the KG branch behaved as it did.

The project is therefore positioned as a reproducible investigation into KG-augmented VQA, with value in its implementation, controls, diagnostics, and limitations rather than in a positive accuracy gain.
