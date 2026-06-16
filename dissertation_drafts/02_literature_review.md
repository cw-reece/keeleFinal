# Literature Review

## 1. Chapter Purpose

This chapter reviews literature relevant to knowledge-augmented Visual Question Answering (VQA), with emphasis on outside-knowledge VQA, structured commonsense resources, knowledge graph retrieval, and fusion strategies. The purpose is not to provide an exhaustive survey of multimodal learning. Instead, the chapter identifies the design pressures that shaped this project: the need for a stable VQA baseline, bounded and inspectable external knowledge retrieval, cautious late fusion, and controlled evaluation.

The project investigates whether bounded, task-specific ConceptNet knowledge graph augmentation, integrated via late fusion, improves OK-VQA validation VQA-soft accuracy compared with a frozen ViLT baseline. The literature is therefore reviewed from a system-development perspective. Each major theme is connected to a practical design decision in the implemented system.

## 2. Visual Question Answering and the Need for Knowledge

Visual Question Answering asks a model to answer a natural-language question about an image. The original VQA benchmark introduced the task as a way to test joint image and language understanding (Antol et al., 2015). A typical VQA system must identify relevant visual content, interpret the question, and produce an answer in natural language or from a fixed answer set. This appears straightforward, but VQA is difficult because correct answers may depend on object recognition, spatial reasoning, attributes, commonsense, and world knowledge.

Early VQA work also revealed a persistent problem: models can achieve plausible accuracy by exploiting language priors and dataset biases rather than by grounding answers in the image. For example, certain question templates often correlate with frequent answers. A model may learn that questions beginning with “What colour is the sky?” are often answered with “blue”, even if it is not using image evidence properly. This matters for knowledge-augmented VQA because an external knowledge module can appear helpful if it shifts predictions toward frequent dataset answers, even when the retrieved knowledge is not actually relevant.

Later datasets such as Visual7W and VCR increased the pressure for grounded reasoning and visual commonsense (Zhu et al., 2016; Zellers et al., 2018). However, the broader challenge remained: answering correctly does not necessarily prove that the system used the right evidence. For this reason, knowledge-augmented systems need more than headline accuracy. They need intermediate artefacts, ablations, and controls that help explain whether knowledge was retrieved, whether it was relevant, and whether it influenced the answer.

### Implication for this project

This project treats VQA as a controlled system-development problem rather than only a benchmark-optimisation task. The system logs retrieved knowledge slices, records configuration-linked runs, and compares fused predictions against a frozen baseline. This design responds directly to the attribution problem in VQA: it is not enough to report an answer; the system must make it possible to inspect what knowledge was retrieved and how evaluation changed when the KG branch was enabled or disabled.

## 3. Outside-Knowledge VQA

Outside-Knowledge Visual Question Answering focuses on questions that cannot be answered reliably from pixels alone. OK-VQA was introduced to evaluate VQA systems that require external knowledge about objects, events, functions, attributes, and commonsense associations (Marino et al., 2019). A-OKVQA further develops this direction by broadening world-knowledge requirements and supporting evaluation formats that can make reasoning more controlled (Schwenk et al., 2022).

These datasets expose cases where a strong visual model may recognise the objects in an image but still fail to answer the question. For example, recognising a stove does not automatically answer what it is used for; recognising a flag does not automatically identify the country; recognising a tool does not automatically explain its function. Outside-knowledge VQA therefore requires some connection between image/question content and world knowledge.

Fact-based VQA also contributed to this area by linking visual questions to supporting facts. FVQA is important because it made explicit the role of supporting evidence in knowledge-based VQA (Wang et al., 2017). In a fully evidence-grounded system, the answer is not only predicted but also connected to a fact or relation. This is useful for evaluation and interpretability, although building such systems increases engineering complexity because the model must retrieve, rank, and use relevant facts.

The key difficulty is that outside knowledge introduces several possible failure points. A system may fail to identify the relevant visual entity. It may identify the entity but link it to the wrong knowledge concept. It may retrieve a relevant fact but fail to connect that fact to the answer vocabulary. Or it may retrieve irrelevant facts that damage the prediction. Therefore, knowledge augmentation is not automatically beneficial; it can also introduce noise.

### Implication for this project

The project uses OK-VQA because it is explicitly knowledge-intensive, but it does not assume that adding ConceptNet will automatically improve accuracy. The implementation is designed to test the knowledge branch under controlled conditions. The final evaluation therefore asks not only whether accuracy improves, but also how different fusion strategies behave when external knowledge is noisy or weakly aligned with the answer space.

## 4. Knowledge Sources for VQA

Knowledge-augmented VQA systems require a source of external knowledge. Possible sources include commonsense knowledge graphs, encyclopaedic knowledge graphs, lexical resources, scene graphs, captions, retrieved text, and memory-based models. The choice of knowledge source affects engineering complexity, retrieval quality, interpretability, and reproducibility.

### 4.1 ConceptNet

ConceptNet is an open commonsense knowledge graph containing labelled edges between natural-language concepts (Speer, Chin and Havasi, 2017). It includes relations such as UsedFor, HasProperty, CapableOf, IsA, PartOf, AtLocation, and RelatedTo. These relation types are relevant to many commonsense questions in OK-VQA, especially questions involving function, affordance, category, properties, and everyday associations.

ConceptNet is attractive for an MSc system-development project because it can be processed into a local store, queried reproducibly, and inspected as triples. It supports auditable retrieval: selected facts can be logged and shown during error analysis. This is useful because the project aims to evaluate whether external knowledge helps and why it may fail.

However, ConceptNet also has limitations. It is noisy, uneven, and contains many generic high-degree concepts. If retrieval is unconstrained, a query can return many weakly relevant facts. Concepts are represented as natural-language phrases, so mapping question tokens to ConceptNet nodes can be brittle. For example, surface-form variation, pluralisation, synonyms, and ambiguous words can all affect retrieval. A question may also require visual concepts that are not explicitly mentioned in the text.

The literature therefore suggests that ConceptNet should not be used as an unrestricted source of facts. It needs relation filtering, hop limits, top-k selection, scoring, and caching. Without these constraints, the KG branch may add irrelevant evidence and reduce accuracy.

### 4.2 Wikidata and DBpedia

Wikidata and DBpedia provide encyclopaedic knowledge about named entities, properties, locations, people, objects, and events (Vrandečić and Krötzsch, 2014). They are useful for questions requiring factual or entity-specific information. For example, questions involving landmarks, brands, countries, or named people may benefit from encyclopaedic sources more than from commonsense sources.

The engineering challenge is that encyclopaedic KGs require stronger entity linking and disambiguation. A visual question may mention “apple”, which could mean a fruit or a company. A system using Wikidata must decide which entity is intended and which properties are relevant. This can dominate project scope. Unrestricted expansion over encyclopaedic KGs can also create very large candidate sets.

### 4.3 WordNet and lexical resources

WordNet provides lexical relations such as synonyms, hypernyms, and hyponyms. It is less suitable as a primary source of commonsense facts, but it can help with normalisation and linking. For example, a lexical resource can help connect “bicycle” and “bike” or identify category-level relations.

For a bounded system-development project, lexical resources are best treated as supporting tools rather than as the main knowledge substrate. They can improve entity matching but do not provide the same range of commonsense relations as ConceptNet.

### 4.4 Visual Genome and scene graphs

Visual Genome provides object, attribute, and relation annotations that connect language and vision (Krishna et al., 2017). Scene graph resources can help identify visual entities and relations that are not obvious from question text alone. This is important because OK-VQA questions often depend on objects or context in the image.

However, using scene graphs introduces additional dependencies. The system either needs ground-truth scene graph annotations or a reliable scene graph generator. That can make evaluation more complex and may shift the project from KG augmentation into object detection and scene graph modelling.

### 4.5 Captions and retrieved text

Caption-based methods use generated or retrieved captions as a bridge between images and text-based knowledge. Recent knowledge-based VQA work has argued that captions can improve retrieval by summarising visual content in language, making it easier to query external knowledge sources (Feng et al., 2024). Captions can provide visual context not present in the question, such as objects, scene type, activity, and attributes.

Captions are attractive because they provide a language representation of the image. However, they also introduce errors. A caption generator may omit the object needed for the question or hallucinate content. If the KG retrieval depends on faulty captions, the knowledge branch may retrieve irrelevant facts.

### Implication for this project

The project selects ConceptNet as the main knowledge source because it is locally deployable, inspectable, and aligned with commonsense OK-VQA questions. Wikidata, Visual Genome, and caption-based retrieval are discussed as future extensions rather than core dependencies. This keeps the implementation bounded and allows the project to focus on controlled ConceptNet slicing and late fusion.

## 5. Knowledge Integration Strategies

Once external knowledge is retrieved, a VQA system must integrate it into prediction. The literature includes several broad approaches: retrieval-augmented integration, early fusion, late fusion, hybrid symbolic-neural reasoning, graph reasoning, and memory/prompting-based methods.

### 5.1 Retrieval-augmented knowledge integration

Retrieval-augmented VQA systems retrieve candidate facts, passages, or triples and condition the answer process on this evidence. This makes knowledge use more explicit than relying only on model parameters. However, retrieval quality is critical. If the retrieved evidence is irrelevant, the system may be distracted or harmed.

The main engineering problem is therefore not only how to retrieve more knowledge, but how to retrieve less irrelevant knowledge. A retrieval module must identify candidate entities, map them to a knowledge source, expand relevant relations, and rank results. Each step can introduce noise.

### 5.2 Early fusion

Early fusion methods combine visual, textual, and knowledge representations before or during deep model encoding. Concept-aware models such as ConceptBERT represent this direction by integrating concept-level information into the representation process (Gardères et al., 2020). Early fusion can be powerful because the model can learn interactions between image, question, and knowledge at a deep level.

The drawback is reduced modularity and auditability. If knowledge is deeply entangled inside a large model, it becomes harder to determine whether an accuracy change came from useful knowledge, training dynamics, or representation learning. Early fusion may also require more compute and training data than is appropriate for a bounded MSc build.

### 5.3 Late fusion

Late fusion keeps the baseline VQA branch and the knowledge branch separate until the final answer scoring stage. A baseline model produces answer logits, while a knowledge branch produces an additional answer signal. These are then combined through a weighted, gated, or otherwise constrained mechanism.

Late fusion has several advantages for a system-development dissertation. It allows the baseline to be frozen. It supports direct ablations, because the KG branch can be enabled, disabled, shuffled, or replaced. It also allows separate inspection of baseline predictions and KG-derived evidence.

The limitation is that late fusion can only help if the KG-derived signal is relevant and calibrated. If the KG branch produces noisy logits, weighted fusion can damage the baseline. This means late fusion requires careful evaluation and may need gating or top-N constraints.

### 5.4 Hybrid symbolic-neural reasoning

KRISP is an important example of hybrid symbolic-neural VQA for knowledge-based questions (Marino et al., 2021). It integrates implicit visual-language knowledge with symbolic knowledge, showing that combining different knowledge types can be beneficial. KRISP is relevant because it demonstrates the value of separating implicit model knowledge from explicit symbolic knowledge.

However, full hybrid systems can be complex to reproduce. They may require specific feature extractors, symbolic modules, graph construction steps, and training pipelines. For this project, KRISP is more important as a design influence than as a direct implementation target. It motivates the separation between a baseline model and an explicit knowledge branch.

### 5.5 Graph-based reasoning

Graph-based approaches build structured representations over objects, relations, and knowledge facts. TRiG, for example, applies transformer reasoning over graphs for VQA (Gao et al., 2022). Multi-modal semantic graph approaches also represent image and knowledge content as structured graphs (Jiang and Meng, 2023).

These approaches show that graph structure can support reasoning when the graph is relevant and well-constructed. However, they also highlight a major risk: graph quality matters. If the graph slice contains irrelevant nodes or misses the key relation, the reasoning model has little chance of producing the correct answer. Bigger graphs are not automatically better; they may add noise.

### 5.6 Memory and prompting-based alternatives

More recent systems may use large models, prompting, or memory-based retrieval. These can be powerful, but they introduce different methodological issues. Prompting-based systems may be harder to reproduce exactly, especially if they depend on external APIs or changing model versions. Large models also make attribution difficult because much knowledge is already stored implicitly in model parameters.

For this project, those approaches are treated as outside the core scope. The aim is not to build the strongest possible VQA model using the largest available model. The aim is to build an auditable KG-augmented system where retrieved knowledge is explicit and evaluation is controlled.

### Implication for this project

The project uses late fusion because it provides the best balance between feasibility, modularity, and evaluation control. Weighted fusion directly tests whether KG evidence helps. Gated fusion tests whether the model can suppress unreliable knowledge. Top-N constrained fusion limits KG influence to plausible baseline answers. This design follows the literature’s warning that external knowledge can help only when retrieval and fusion are controlled.

## 6. Task-Specific Knowledge Graph Slicing

A central challenge in KG-augmented VQA is selecting the right subgraph. A full KG is far too large and noisy to inject directly. The system must build a task-specific slice: a small set of facts likely to be relevant to the image-question pair.

A typical KG slicing pipeline includes:

1. Extract entities from the question, image detections, captions, or scene graph.
2. Map those entities to KG concepts.
3. Expand neighbours within a limited hop depth.
4. Filter relation types.
5. Score candidate facts for relevance.
6. Select a bounded top-k set.
7. Cache the result for reproducibility.

The literature suggests that slice quality is often more important than slice size. Adding more facts may increase coverage, but it also increases noise. A small, relevant slice is more useful than a large, generic neighbourhood.

Several design choices affect slice quality. Relation filtering can remove unhelpful edges. Hop limits prevent uncontrolled expansion. Top-k selection keeps the slice inspectable. Degree penalties can reduce the influence of generic hubs. Caption or visual-object grounding can improve entity selection, although it adds complexity.

In this project, the slice builder uses question-derived entities, ConceptNet relation filtering, hop depth, top-k limits, neighbour limits, and configuration-aware caching. This makes the slice bounded and reproducible. The random-slice control further tests whether any observed effect comes from task-specific retrieval or merely from the presence of a KG branch.

### Implication for this project

The project treats KG slicing as a primary implementation component rather than as a minor preprocessing step. The negative final result reinforces the literature’s warning: bounded slicing improves auditability and safety, but it does not guarantee that retrieved facts will align with the answer required by OK-VQA. Future improvements should therefore focus on better grounding, slice scoring, and fact-answer alignment.

## 7. Evaluation Challenges in Knowledge-Augmented VQA

Evaluation of knowledge-augmented VQA is difficult because accuracy alone does not prove knowledge use. A model can answer correctly for the wrong reason, and a knowledge branch can produce apparent gains through regularisation, reranking, or dataset priors.

A strong evaluation should therefore include:

- a stable baseline;
- matched comparison between baseline and knowledge-augmented variants;
- ablations over key knowledge settings;
- controls using random or shuffled knowledge;
- logged retrieved evidence;
- qualitative error analysis;
- careful separation of headline full-validation results from smaller diagnostic experiments.

OK-VQA uses VQA-style soft accuracy, which gives partial credit based on agreement with human answers. This is appropriate for VQA, but it introduces challenges. Short answer strings must be normalised, and semantically reasonable answers may receive low credit if they do not match annotator wording. This is particularly important when KG facts suggest a semantically related but not exact answer.

Robustness literature also warns that shortcut learning can distort interpretation (Peng and Li, 2024). If a model exploits question priors, an apparent knowledge gain may not reflect genuine external reasoning. For this reason, random-slice or shuffled-edge controls are important. If random knowledge improves performance, the gain should not be attributed to meaningful knowledge retrieval.

### Implication for this project

The project evaluation includes a frozen baseline, full-validation fusion comparisons, a 512-example ablation matrix, and a random-slice control. The random-slice control is especially important because it tests whether unrelated KG evidence can create a spurious gain. In the final result, random-slice fusion did not improve over the frozen baseline, which strengthens the attribution argument.

## 8. Synthesis: Design Rationale from the Literature

The literature supports four major design decisions in this project.

First, the system needs a frozen baseline. VQA models can vary significantly across training runs, and knowledge-augmented systems can be difficult to interpret if the baseline is not stable. Freezing the baseline allows matched comparison.

Second, external knowledge must be bounded and inspectable. ConceptNet is useful because it provides commonsense relations, but it is noisy and contains generic hubs. The system therefore uses bounded slicing, relation filtering, top-k selection, and cache keys.

Third, fusion should be modular and controlled. Early fusion can be powerful but difficult to attribute. Late fusion allows the KG branch to be enabled, disabled, constrained, or replaced. This supports ablation and random-slice controls.

Fourth, evaluation must include controls. A random-slice control tests whether any gain comes from relevant knowledge or merely from the extra branch. Even though the final result was negative, the control improves the credibility of the evaluation.

## 9. Relationship to the Implemented System

The implemented system follows the literature-derived rationale in the following way.

The OK-VQA baseline uses a ViLT-based answer classifier over a fixed answer vocabulary. This provides the frozen comparison point.

The ConceptNet branch extracts entities from question text, retrieves ConceptNet neighbours, filters relations, ranks candidate facts, and selects a bounded top-k slice. This implements the literature recommendation that KG retrieval should be constrained and inspectable.

The knowledge encoder converts selected facts into a KG-derived answer signal. The fusion module then combines this signal with baseline logits through weighted, gated, or top-N constrained fusion. This implements the late-fusion design chosen for modularity and controlled evaluation.

The final evaluation uses full-validation runs, ablations, and a random-slice control. This responds to the evaluation literature’s concern that knowledge gains may be spurious unless tested against controls.

## 10. Literature-Informed Expectations

Based on the literature, the project did not assume that ConceptNet augmentation would automatically improve accuracy. Instead, three possible outcomes were expected.

The first possible outcome was improvement: task-specific ConceptNet slices might provide useful external evidence and improve VQA-soft accuracy over the frozen baseline.

The second possible outcome was no measurable gain: retrieved facts might be plausible but not sufficiently aligned with the short answer vocabulary or the model’s decision process.

The third possible outcome was degradation: noisy or poorly calibrated KG evidence might perturb baseline logits and reduce accuracy.

The final results matched the second and third outcomes rather than the first. Naive weighted fusion degraded performance substantially, constrained weighted fusion reduced the harm to a near-zero negative delta, and gated fusion preserved baseline performance. This result is consistent with the literature’s warning that external knowledge is only useful when retrieval, grounding, and fusion are well aligned.

## 11. Chapter Summary

This literature review shows that knowledge-augmented VQA is difficult because correct answers may require external knowledge, but external knowledge can also introduce noise. OK-VQA motivates the need for world knowledge, while ConceptNet provides a practical commonsense source for a bounded system-development project. However, ConceptNet requires careful slicing, relation filtering, and evaluation controls.

The reviewed literature supports the project’s main design choices: a frozen baseline, bounded ConceptNet slicing, late fusion, top-N constraints, gated fusion, run logging, and a random-slice control. The literature also prepares the interpretation of the final result. The fact that KG augmentation did not improve accuracy does not invalidate the project; rather, it demonstrates a known challenge in knowledge-augmented VQA: retrieved knowledge must be relevant, grounded, answer-aligned, and calibrated before it can reliably improve prediction.

## References

Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.L. and Parikh, D. (2015) ‘VQA: Visual Question Answering’, in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2425–2433.

Feng, Y., Luo, C., Liu, J., Zheng, H., Dai, W., Shen, Z., Ma, C., Qiao, Y. and Wang, C. (2024) ‘Caption matters: a new perspective for knowledge-based visual question answering’, *Neural Networks*, 173, pp. 1–13. doi:10.1016/j.neunet.2024.01.027.

Gao, P., Zheng, C., Wang, R., Li, J., Qiao, L. and Li, H. (2022) ‘TRiG: Transformer Reasoning on Graphs for Visual Question Answering’, in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 15583–15592.

Gardères, F., Ziaeefard, M., Abeloos, B. and Lécué, F. (2020) ‘ConceptBERT: Concept-Aware Representation for Visual Question Answering’, in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 489–498.

Jiang, L. and Meng, Z. (2023) ‘Knowledge-Based Visual Question Answering Using Multi-Modal Semantic Graph’, *Electronics*, 12(6), 1390. doi:10.3390/electronics12061390.

Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D.A., Bernstein, M. and Fei-Fei, L. (2017) ‘Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations’, *International Journal of Computer Vision*, 123(1), pp. 32–73. doi:10.1007/s11263-016-0981-7.

Li, S., Gong, C., Zhu, Y., Luo, C., Hong, Y. and Lv, X. (2024) ‘Context-aware Multi-level Question Embedding Fusion for Visual Question Answering’, *Information Fusion*, 102, 102000. doi:10.1016/j.inffus.2023.102000.

Lymperaiou, M. and Stamou, G. (2024) ‘A survey on knowledge-enhanced multimodal learning’, *Artificial Intelligence Review*, 57, 284. doi:10.1007/s10462-024-10825-z.

Marino, K., Rastegari, M., Farhadi, A. and Mottaghi, R. (2019) ‘OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge’, in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

Marino, K., Rastegari, M., Farhadi, A. and Mottaghi, R. (2021) ‘KRISP: Integrating Implicit and Symbolic Knowledge for Open-Domain Knowledge-Based Visual Question Answering’, in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 14111–14120.

Peng, D. and Li, Z. (2024) ‘Robust visual question answering via polarity enhancement and contrast’, *Neural Networks*, 106560. doi:10.1016/j.neunet.2024.106560.

Schwenk, D., Khandelwal, A., Clark, C., Marino, K. and Mottaghi, R. (2022) ‘A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge’, in *Proceedings of the European Conference on Computer Vision (ECCV)*.

Speer, R., Chin, J. and Havasi, C. (2017) ‘ConceptNet 5.5: An Open Multilingual Graph of General Knowledge’, in *Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (AAAI)*, pp. 4444–4451.

Vrandečić, D. and Krötzsch, M. (2014) ‘Wikidata: a free collaborative knowledgebase’, *Communications of the ACM*, 57(10), pp. 78–85. doi:10.1145/2629489.

Wang, P., Wu, Q., Shen, C., Dick, A. and Van den Hengel, A. (2017) ‘FVQA: Fact-Based Visual Question Answering’, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(10), pp. 2413–2427. doi:10.1109/TPAMI.2017.2754226.

Wu, J., Wang, R., Li, J., Qiao, L., Gao, P. and Li, H. (2022) ‘MAVEx: Memo-Aware Visual Question Answering’, in *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(3), pp. 2943–2951.

Yan, X., Chang, X., Luo, C., Sun, Q., Liu, J., Qiao, Y. and Peng, Y. (2024) ‘Knowledge-aware image understanding with multi-level visual representation enhancement for visual question answering’, *Machine Learning*, 113, pp. 3789–3805. doi:10.1007/s10994-024-06565-w.

Yang, Z., Liu, X., Meng, F., Zhang, J., Chen, J. and Gao, J. (2022) ‘PICa: Plug-and-Play Image Captioning Model for TextVQA’, in *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(3), pp. 2921–2929.

Zellers, R., Bisk, Y., Schwartz, R. and Choi, Y. (2018) ‘From Recognition to Cognition: Visual Commonsense Reasoning’, arXiv preprint arXiv:1811.10830.

Zhu, Y., Groth, O., Bernstein, M. and Fei-Fei, L. (2016) ‘Visual7W: Grounded Question Answering in Images’, in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4995–5004.
