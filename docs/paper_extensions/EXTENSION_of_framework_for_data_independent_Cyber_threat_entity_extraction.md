## EXTENSION TO THE FOLLOWING PAPER AT THE END...

## Towards a scalable AI-driven framework for data-independent Cyber Threat Intelligence Information Extraction

1st Olga Sorokoletova<br>
Data & Artificial Intelligence — CY4GATE S.p.A. — Rome, Italy<br>
olga.sorokoletova@cy4gate.com

2nd Emanuele Antonioni<br>
Data & Artificial Intelligence — CY4GATE S.p.A. — Rome, Italy<br>
emanuele.antonioni@cy4gate.com

3rd Giordano Colò<br>
Data & Artificial Intelligence — CY4GATE S.p.A. — Rome, Italy<br>
giordano.colo@cy4gate.com

### Abstract
Cyber Threat Intelligence (CTI) is critical for mitigating threats to organizations, governments, and institutions, yet the necessary data are often dispersed across diverse formats. AI-driven solutions for CTI Information Extraction (IE) typically depend on high-quality, annotated data, which are not always available. This paper introduces 0-CTI, a scalable AI-based framework designed for efficient CTI Information Extraction. Leveraging advanced Natural Language Processing (NLP) techniques, particularly Transformer-based architectures, the proposed system processes complete text sequences of CTI reports to extract a cyber ontology of named entities and their relationships.

Our contribution is the development of 0-CTI, the first modular framework for CTI Information Extraction that supports both supervised and zero-shot learning. Unlike existing state-of-the-art models that rely heavily on annotated datasets, our system enables fully dataless operation through zero-shot methods for both Entity and Relation Extraction, making it adaptable to various data availability scenarios. Additionally, our supervised Entity Extractor surpasses current state-of-the-art performance in cyber Entity Extraction, highlighting the dual strength of the framework in both low-resource and data-rich environments.

By aligning the system’s outputs with the Structured Threat Information eXpression (STIX) format, a standard for information exchange in the cybersecurity domain, 0-CTI standardizes extracted knowledge, enhancing communication and collaboration in cybersecurity operations.

**Index Terms**: Cyber Threat Intelligence, Natural Language Processing, Structured Threat Information Expression, Named Entity Recognition, Relation Extraction

---

### I. INTRODUCTION
Cyber Threat Intelligence (CTI) [1] is a fundamental discipline in cybersecurity that focuses on the collection, analysis, and interpretation of threat data concerning cyberattacks. This process involves collecting data from various sources, such as dark web forums, security incident databases, and network sensors, to build a comprehensive picture of potential threats. In this context, a key role is played by the Cyber Threat Intelligence analyst: a professional specializing in the collection, analysis, and interpretation of cyber threat intelligence. The CTI analyst identifies Indicators of Compromise (IOCs), analyzes attackers’ Techniques, Tactics, and Procedures (TTPs), and provides strategic recommendations to improve security. Additionally, the CTI analyst collaborates with other security teams to implement preventive measures and respond to incidents, contributing to a secure digital environment.

Despite several technological advances, the work of the CTI analyst still requires many manual knowledge extraction steps, such as data collection, verification, and analysis, which are laborious and time-consuming. Moreover, many companies, organizations, or public institutions lack the resources to hire a dedicated CTI analyst, exposing themselves to various risks, including vulnerability to cyber attacks, loss of sensitive data, reputational damage, and potential breaches of data security regulations. The absence of a CTI expert can significantly undermine an organization's ability to defend itself against cyber threats. To address these problems, the industry has been investing heavily in the integration of Artificial Intelligence (AI) tools [2], [3]. AI enables the automation of many manual tasks, such as network monitoring and data analysis, increasing the efficiency and speed of the intelligence process. For example, Machine Learning (ML) algorithms can identify patterns and anomalies in data, signaling imminent threats for faster and more accurate responses. Additionally, AI helps predict future threats by analyzing historical and behavioral trends of attackers.

In line with the intention of integrating AI techniques effectively within the CTI analyst’s workflow, this paper introduces 0-CTI, an innovative system designed to address various challenges in the field of Cyber Threat Intelligence. The system employs a fully Machine Learning-based approach to extract knowledge in a structured format from unstructured natural language texts. 0-CTI was developed to significantly accelerate the work of CTI analysts by automating numerous laborious steps in the data extraction and analysis process. Traditionally, analysts spend considerable time manually collecting, verifying, and interpreting threat information. With 0-CTI, these tasks can be quickly and accurately performed, enabling analysts to focus on more strategic and decision-making responsibilities.

Another major advantage of 0-CTI is its ability to support organizations lacking a dedicated CTI analyst. Automating knowledge extraction steps allows these organizations to mitigate risks associated with the absence of specialized expertise. Furthermore, 0-CTI generates knowledge in a STIX-compliant format, which is a widely recognized standard in the community. This facilitates universal information sharing, improving collaboration between different organizations and strengthening the collective ability to defend against cyber threats. Thanks to compliance with STIX, extracted information can be easily integrated into existing systems, enhancing the effectiveness of cybersecurity operations.

0-CTI represents a remarkable technological advancement in CTI, harnessing the potential of Transformer neural networks to achieve superior performance in state-of-the-art Entity and Relation Extraction. By utilizing these advanced networks, 0-CTI can accurately recognize and analyze a wide range of entities and their interconnections, overcoming limitations of existing solutions, such as the limited number of classes and strong data dependence. In addition to the traditional NER system based on prior dataset training, our system proposes a zero-shot variant of the Entity Extraction model. This variant employs single-level taxonomies of STIX-compliant classes combined with zero-shot Named Entity Recognition (NER) systems like GLiNER [4]. 0-CTI can recognize and categorize new or unknown entities without requiring a preliminary training phase, making the integration and update process much faster and more flexible. Additionally, 0-CTI employs a novel algorithm based on cross-encoder neural networks to extract relationships between entities in zero-shot mode. This approach allows the system to identify connections between different entities without specific training, further expanding its capabilities in analyzing and correlating information. The final result is a knowledge graph that embeds all relevant information expressed by the original plain text.

With these features, 0-CTI can seamlessly scale to diverse applications, independent of an organization’s capacity to deploy specialists for data preparation and validation. This means that even organizations without dedicated CTI resources can benefit from an advanced, automated system for threat extraction and analysis, significantly improving their cybersecurity awareness.

---

### II. INFORMATION EXTRACTION OF STIX OBJECTS
Structured Threat Information eXpression (STIX) was introduced as a language for the systematic and machine-readable representation of CTI data in a publication [5] authored by the MITRE Corporation, a research organization distinguished for noteworthy contributions to the field of Cybersecurity. Acknowledged as a widely adopted standard within the CTI community, STIX has established a systematic classification covering aspects of suspicion, compromise, and attribution. With objects and descriptive relationships, this classification enables consistent data-sharing among organizations.

Formally, STIX can be characterized as a schema that defines a taxonomy of Cyber Threat Intelligence, comprising six distinct classes, three of which are the primary focus here:
- STIX Domain Objects (SDOs): higher-level intelligence objects representing behaviors and constructs that threat analysts engage with to comprehend the threat landscape.
- STIX Cyber-observable Objects (SCOs): encapsulate Indicators of Compromise (IOCs), a perennial concern in Cybersecurity.
- STIX Relationship Objects (SROs): connect SDOs together, SCOs together, and SDOs with SCOs.

Given that this work addresses the Information Extraction problem, it is crucial to align the formulation with STIX terminology to standardize system output, facilitating efficient information sharing within the cybersecurity community.

The task of Information Extraction from textual data, as outlined by Piskorski and Yangarber in [6], involves automating the identification and extraction of structured information or knowledge from unstructured text. It encompasses the identification and extraction of specific entities, relationships, and events mentioned in the text, transforming raw data into a structured format that is easily processable and analyzable. In our case, the holistic task breaks down into two subtasks, which can be tackled either concurrently or sequentially: Entity Extraction (EE) and Relation Extraction (RE). Within the framework of STIX, entities for Entity Extraction are represented by a subset of SDOs and SCOs, while relationships in Relation Extraction are defined by SROs.

---

### III. RELATED WORKS
The necessity to automate Cyber Threat Intelligence data processing, emphasizing Entity and Relation Extraction in compliance with community standards, has driven researchers to embrace Information Extraction methodologies in Cybersecurity. Despite significant efforts (Gasmi et al. 2019 [7]; Legoy et al. 2020 [8]), many current methods struggle with the volume of extracted entities and relations and do not fully adhere to the STIX taxonomy. However, the adoption of Artificial Intelligence (AI), particularly Transformer models, is enhancing the efficiency of these processes, facilitating more robust and automated CTI analysis.

Weerawardhana et al. (2015) [9] compared a Machine Learning-based approach and a Part-of-Speech tagging method for Information Extraction, focusing on vulnerability databases. Li et al. (2019) [10] introduced a self-attention-based Neural Network for Cybersecurity NER, integrating features from words and characters using CNN and a self-attention mechanism based on BiLSTM and CRF models.

Ranade et al. (2021) [11] developed CyBERT, a domain-specific BERT model fine-tuned on a cybersecurity corpus, further improved by retraining on a corpus tailored to STIX entities. This model outperformed others in the Massive Text Embedding Benchmark (MTEB) Leaderboard (Muennighoff et al., 2022 [12]).

Wang et al. (2023) [13] presented an Entity Recognition model that uses BERT for dynamic word vectors and BiLSTM-CRF for word sequence encoding, refined with CTI-specific knowledge engineering. Alam et al. (2022) [14] introduced CyNER, an open-source Python library for cybersecurity NER, utilizing Transformer-based models and heuristics for extracting IOCs.

Zhou et al. (2022) [15] developed CTI View, an automated system for extracting and analyzing unstructured CTI associated with Advanced Persistent Threats (APTs), employing a Text Extraction framework and a BERT-BiLSTM-CRF model enhanced with a GRU layer. Zhou et al. (2023) [16] later presented CDTier, a CTI dataset emphasizing threat Entity and Relation Extraction, improving model accuracy in extracting knowledge objects and relationships.

Marchiori and Conti (2023) [17] introduced STIXnet, a modular solution for extracting STIX Objects from CTI Reports. STIXnet, aligned with the STIX taxonomy, represents the state-of-the-art in Information Extraction within the domain, but our system demonstrates superior performance in Entity and Relation Extraction compared to STIXnet. Unlike STIXnet’s rule-based approach, 0-CTI leverages Transformers for Entity Extraction and a novel cross-encoder model for Relation Extraction, employing a dataless training approach.

Our zero-shot NER model, based on GLiNER [4], addresses shortcomings identified in previous studies [18]–[24], such as slower processing speeds, large parameter sizes, and limited capability to predict multiple entity types concurrently. In conclusion, 0-CTI stands out as a unique system capable of simultaneous Entity and Relation Extraction, leveraging advanced AI-based methodologies to identify a comprehensive range of STIX-compliant entities and relationships, operating in a dataless manner, and advancing the state-of-the-art in CTI Entity Extraction.

---

### IV. 0-CTI SYSTEM
0-CTI is a modular, Transformers-based system that serves as a “lens” applied to raw, unstructured collections of CTI, zooming in on STIX Domain Objects, IOCs, and their relationships expressed as STIX Relationship Objects. The overall architecture is delineated in a block scheme (Fig. 1), with a detailed explanation of each component provided here.

#### A. Dataset and Text Processing
In the initial phase, the Text Processing module ingests CTI documents and processes the raw textual content, producing a dataset containing sanitized, tokenized, and labeled text chunks. This dataset is the output of this module and the input for subsequent modules.

The system accepts CTI documents in English in various formats, including PDF, DOCX, or HTML. The extraction process identifies 9 entity classes corresponding to 9 STIX Domain Objects, and 21 relation classes corresponding to 21 STIX Relationship Objects. The dataset used for training the supervised NER model contained annotations for entities (statistics in Table I), but did not include annotations for relations, which are:
AUTHORED BY, ATTRIBUTED TO, BEACONS TO, COMMUNICATES WITH, COMPROMISES, CONSISTS OF, CONTROLS, DELIVERS, DOWNLOADS, DROPS, EXFILTRATE TO, EXPLOITS, HAS, HOSTS, IMPERSONATES, INDICATES, LOCATED AT, ORIGINATES FROM, OWNS, TARGETS, USES.

The Identity SDO is categorized to distinguish between persons and organizations, while the Indicator SDO is subdivided to encompass 23 types of IOCs/SCOs within the dataset, including domain names, email addresses, file hashes, IP addresses, URLs, and registry key paths, among others (Table II).

The IOC-finder discovers these indicators. Concurrently, the input presented to the ML-driven core of Entity Extraction undergoes a masking process using strings associated with respective IOC types to prevent the learning process from being disrupted by erratic tokenization.

Aggregating data from diverse sources introduces noise, necessitating comprehensive preprocessing prior to training. Our text processing workflow has three stages:
- Standardization and artifact removal (eliminate log file names, redundant separators, unsupported characters; sanitize text; handle IOC tables, images, and code snippets embedded in reports).
- Chunking and recalibration (recursive chunkizer with overlap to capture distant relationships; recalibrate entity offsets).
- CoNLL formatting (produce outputs aligned with CoNLL-2003 conventions).

Note: The STIX taxonomy comprises 18 SDOs; however, some are not pertinent to the NER task and instead serve as workaround instruments (e.g., Grouping SDO). [^1]

#### B. Entity Extraction with Supervised NER and IOC-finder
The Entity Extraction module consists of two primary submodules: the core extractor and the IOC-finder. The core extractor includes a leading Transformer model for token classification fine-tuned on an annotated dataset, and a zero-shot NER model (GLiNER). The IOC-finder is a mandatory sub-component, implemented using Floyd Hightower’s IOC Finder (regex patterns). The rationale: IOCs are highly structured, enabling efficient detection through pattern matching. IOCs are often listed in tables without context; some (e.g., hashes) are high-entropy and degrade tokenization and Transformer training if left unmasked.

For supervised NER in the core, we employ a model from the BAAI General Embedding (BGE) family [25]. BGE includes BERT-based embedding models across scales (large/base/small), available for Chinese and English. BERT-based models are strong performers for NER [^3], and BGE offers a good balance of efficiency and representation quality. We use `BAAI/bge-base-en-v1.5` as the backbone Transformer. [^4]

The confidence score of the fine-tuned BGE-based model is used to merge extractions with those of the zero-shot extractor in applications where annotated data are available. Prior to merging, we threshold each NER output to discard false positives.

After the Entity Extraction submodules complete, their outputs are integrated with Text Processing outputs, yielding processed texts and extracted entities with positions. This composite is the input to Relation Extraction. We adopt a sequential EE→RE architecture to preserve modularity, acknowledging that RE performance can be impacted when EE misses entities.

#### C. Zero-Shot Components
Zero-shot learning empowers models to predict classes they have never encountered during training, enabling NER without annotated datasets. Another relevant development is zero-shot classification via cross-encoders, which process pairs of sequences together to capture nuanced dependencies—useful for Relation Extraction (RE).

##### 1) Zero-Shot Named Entity Recognition
We perform zero-shot NER by combining GLiNER [4] with a flat taxonomy and a class substitution paradigm. Each target entity class is divided into child categories; if a token is assigned a child label, it is assigned to the parent class (e.g., MALWARE → {Malicious Software, Trojan, Ransomware, …}).

GLiNER is the only available zero-shot entity extractor that does not rely on resource-intensive LLMs, reinforcing the framework’s goal of being ready-to-use even in resource-limited conditions. GLiNER leverages pretrained language models to encode input text and treats extraction as an NLI-style inference task, assessing whether a hypothesis (presence of a specific entity type) is supported by the premise (input text).

This integration enables extending recognized classes without collecting/annotating new data.

##### 2) Relation Extraction using Cross-Encoders
Cross-encoders [26] process sentence pairs jointly and excel at assessing semantic relationships (e.g., entailment). We propose an RE algorithm that integrates EE outputs and adheres to STIX relations:
- Entity Extraction: identify IOCs and entities via IOC-finder and NER (supervised, zero-shot, or both).
- Potential Relations Creation: given entity types and STIX’s SRO tables for type pairs, generate candidate relation sentences of the form “<Entity1> <Relation> <Entity2>.” For example, “APT1 targets Microsoft.” [^5]
- Relation Evaluation: feed original text and candidate relation sentence to a cross-encoder to score the likelihood that the relation holds in context.
- Threshold-Based Validation: accept relations whose scores exceed a threshold to retain high-probability relations.
- Relation Disambiguation: when inverse/ambiguous directions exist, keep the direction with the higher cross-encoder score.

This yields a cyber graph (see Fig. 2). Methodical evaluation and disambiguation produce robust performance across diverse texts.

---

### V. EXPERIMENTAL EVALUATION
Our evaluation focuses on supervised NER performance and qualitative assessment of zero-shot components.

#### A. Performance Evaluation of the Supervised Entity Extractor
We conduct two experiments. First, we compare two Transformer models and a Word2Vec+LSTM on an OpenCTI-sourced dataset. Second, we benchmark the best Transformer model against STIXnet [17].

In both experiments, we identify IOCs via IOC-finder and replace them with placeholders. Given IOC-finder accuracy approaching 100%, we exclude its results from evaluation to focus on context-dependent submodules.

1) Models Comparison on Our Dataset: We perform an 18-class classification task (9 classes with BIO tagging). The dataset comprises 3500 training chunks and 880 test chunks. Some chunks have no entities; we expect empty outputs and aim to minimize false positives.

Training setup for Transformers: 12 epochs, Cross-Entropy loss, AdamW optimizer, batch size 4, learning rate 2e-5. The first network (0-CTI BGE) uses BGE-en-v1.5 embeddings [25]; the second (0-CTI CyBERT) uses cybersecurity-specific embeddings [11]. The LSTM is trained for 200 epochs with Cross-Entropy loss, Adam optimizer, batch size 64, learning rate 1e-3; architecture includes 2 LSTM layers and 2 dense layers (1024 units) before output; Word2Vec pretrained embeddings (GoogleNews-vectors-negative300).

Results (F1 score): both Transformer models achieve strong performance; the best test F1 is 0.85 (0-CTI CyBERT). The LSTM performs substantially worse due to limited contextual capacity.

Per-class observations: Transformers lose generalization on ATTACK PATTERN, CAMPAIGN, IDENTITY PERSON, IDENTITY ORGANIZATION, and TOOL—likely due to dataset imbalance, multiword spans, and ambiguity (e.g., legitimate vs malicious software). Strong performance on THREAT ACTOR, VULNERABILITY, LOCATION, and MALWARE stems from distinctive naming and structured forms (e.g., CVE codes).

2) Comparison on STIXnet Dataset: We fine-tune 0-CTI CyBERT on the STIXnet dataset [17] focused on MITRE ATT&CK groups/threat actors. Using the same preprocessing and training configuration, 0-CTI CyBERT outperforms STIXnet across all classes (overall F1: 0.98 vs 0.77), highlighting the importance of Transformer-based contextual modeling in CTI NER.

#### B. LLM-as-a-Judge Evaluation of Zero-Shot Systems
Large Language Models (LLMs), such as GPT-4, show strong capabilities in understanding and generating unstructured text [27]. The LLM-as-a-Judge method [28] uses LLMs to qualitatively evaluate systems on accuracy, relevance, and coherence.

We use ChatGPT-4 to evaluate zero-shot NER and zero-shot RE on 120 CTI reports. Zero-shot NER achieves an average score of 0.91 (STD 0.06), indicating high and consistent accuracy. Zero-shot RE achieves 0.83 (STD 0.15), reflecting higher task complexity, combinatorial growth of candidate relations, and error propagation from EE to RE in a modular pipeline.

---

### VI. CONCLUSION AND FUTURE WORK
0-CTI automates extraction of critical CTI information, reducing burden on security teams and accelerating access to actionable intelligence. Its scalability and modular design allow data-independence and seamless integration/substitution of submodules, enabling evolution alongside a dynamic cybersecurity landscape—even without specialized expertise or data.

IOC-finder enhances early detection by swiftly discovering IOCs. Advanced NLP (Transformers) drives superior supervised NER performance relative to alternatives for cyber NER. Our novel RE algorithm (cross-encoder, zero-shot) shows promising qualitative results via LLM-as-a-Judge evaluation.

By adhering to STIX, 0-CTI standardizes extracted knowledge and facilitates ecosystem integration with knowledge bases and CTI software, supporting continuous learning and improved operational effectiveness.

Future work includes comprehensive, annotated relation datasets for quantitative RE evaluation and benchmarking, and integrating real-time data sources (e.g., social media, news, dark web) to move from reactive to proactive threat identification.

---

### Appendix — Implementation notes for this repository

This section distills practical steps to implement the paper’s ideas inside this codebase and aligns them with our energy/gating patterns.

- Data preparation
  - Use recursive chunking with overlap to capture relations that span sentences/paragraphs; after chunking, recalibrate entity offsets. See `docs/refs_notes_etc.md/Dynamic_chunking.md` for tunable overlap/size guidance.
  - Pre-mask IOCs (hashes, IPs, URLs, etc.) with stable placeholders before NER to avoid tokenizer fragmentation and training drift; keep a reversible mapping for final rendering.

- Entity Extraction (EE)
  - Zero-shot NER: maintain a flat STIX-compliant taxonomy with subclass synonyms. Calibrate a per-class threshold and optionally merge with supervised NER outputs by confidence.
  - When annotations exist, fine-tune a compact Transformer model for token classification and fuse with zero-shot predictions after per-stream thresholding.

- Relation Extraction (RE) with cross-encoders
  - Candidate generation: enumerate entity pairs permitted by STIX SROs and form templated hypotheses “<E1> <Relation> <E2>” using report-local mentions.
  - Scoring: pass (source text, candidate-hypothesis) pairs to a cross-encoder; keep relations with score ≥ τ, tuned on a small dev set.
  - Disambiguation: for inverse/ambiguous directions, keep the higher-score direction; suppress duplicates across overlapping chunks via non-max suppression on span proximity.

- Coordination with our gating/energy pattern (optional but recommended)
  - Treat “relation acceptance” as an expansion decision. Compute a simple gain (e.g., graph connectivity/consistency increase, or held-out plausibility) minus a cost, and convert to gate open probability via hazard λ(net). Aggregate gains across chunks to avoid over-expansion.
  - Use sparse non-local couplings to nudge acceptance only when it improves global consistency; log hazard_mean and acceptance ratios to verify rare‑but‑impactful behavior.

- STIX graph emission
  - Serialize accepted entities/relations into SDO/SRO/SCO objects; keep stable IDs and include source text spans and confidence scores. Prefer a deterministic pass to reconcile duplicates across overlapping chunks.

- Evaluation and observability
  - Track per-class precision/recall for EE; for RE, report precision at threshold and direction disambiguation accuracy. For zero-shot settings, add a small LLM-as-a-judge qualitative panel to calibrate τ before bulk runs.
  - Enable Delta‑F/η-like progress logs for the acceptance loop if gating is used; audit that expansions correlate with genuine improvements (e.g., fewer contradictions, higher KB coherence).

---

### VII. EXTENSIONS: aPC‑Conditioned Relation Extraction and CODE‑Style Extrapolation

To strengthen zero‑shot performance and out‑of‑distribution robustness, we propose two complementary extensions inspired by recent advances in global polynomial dynamics and arbitrary Polynomial Chaos (aPC) orthonormal bases [29], [30].

- Orthonormal calibration of relation scores (aPC)
  - Map raw cross‑encoder scores s∈[0,1] to ξ=2s−1∈[−1,1] and fit a low‑degree orthonormal polynomial basis Pk(ξ) with respect to the empirical score distribution (aPC or Legendre as a fixed fallback).
  - Use the calibrated curve to (i) set threshold τ that is stable across corpora, (ii) apply margin rules for direction disambiguation, and (iii) reduce overfitting to a single dev set.
  - Practical recipe: start with degree 2–4; re‑estimate basis periodically from held‑out reports to keep calibration current.

- Acceptance as a gated decision with global consistency gain
  - Treat relation acceptance as an expansion decision; compute net = gain − cost, where gain aggregates graph‑consistency signals (e.g., fewer contradictions, improved connectivity) and cost regularizes over‑expansion.
  - Convert net to a one‑step open probability via hazard λ(net) and accept stochastically or by a calibrated threshold on η_gate = 1 − exp(−λ(net)). Log acceptance and hazard_mean for observability.

- CODE‑style extrapolation evaluation
  - Evaluate RE under novel report sources or structures (new “initial conditions”): hold out publishers/time windows and measure precision@τ and direction accuracy “far from” the tuning set.
  - Compare baselines: (i) raw scores + fixed τ, (ii) Legendre‑calibrated scores, (iii) aPC‑calibrated scores. Prefer configurations with fewer threshold backtracks and more stable performance across splits.

These additions retain dataless operation (no new annotations required), sit on top of the existing EE/RE pipeline, and primarily improve conditioning and extrapolation without altering model capacity.

---

### Tables

Table I. Distribution of labeled entities in the dataset.

| Entity Type          | Count |
|----------------------|------:|
| ATTACK PATTERN       |  2993 |
| CAMPAIGN             |   553 |
| IDENTITY ORGANIZATION|  2633 |
| IDENTITY PERSON      |   551 |
| LOCATION             |  6782 |
| MALWARE              | 10902 |
| THREAT ACTOR         |  6228 |
| TOOL                 |  2529 |
| VULNERABILITY        |   786 |
| Total                | 33957 |

Table II. Distribution of labeled IOCs in the dataset.

| IOC Type                              | Count |
|---------------------------------------|------:|
| INDICATOR ATTACK TACTICS ENTERPRISE   |    49 |
| INDICATOR ATTACK TACTICS MOBILE       |     1 |
| INDICATOR ATTACK TECHNIQUES ENTERPRISE|   789 |
| INDICATOR ATTACK TECHNIQUES MOBILE    |     1 |
| INDICATOR BITCOIN ADDRESSES           |   484 |
| INDICATOR CVES                        |   700 |
| INDICATOR DOMAINS                     |  8716 |
| INDICATOR EMAIL ADDRESSES             |   728 |
| INDICATOR FILE PATHS                  |   800 |
| INDICATOR IMPHASHES                   |     5 |
| INDICATOR IPV4S                       |  2049 |
| INDICATOR IPV4 CIDRS                  |    27 |
| INDICATOR MAC ADDRESSES               |     4 |
| INDICATOR MD5S                        |  3816 |
| INDICATOR MONERO ADDRESSES            |     5 |
| INDICATOR REGISTRY KEY PATHS          |   260 |
| INDICATOR SHA1S                       |  1559 |
| INDICATOR SHA256S                     |  6463 |
| INDICATOR SHA512S                     |     4 |
| INDICATOR SSDEEPS                     |     2 |
| INDICATOR TLP LABELS                  |     9 |
| INDICATOR URLS                        |  1609 |
| INDICATOR USER AGENTS                 |    23 |
| Total                                 | 28103 |

---

### Figures
- Fig. 1. 0-CTI pipeline. Orange bold connectors denote mandatory components and a fully dataless flow; orange dashed connectors denote integration of supervised NER; gray connectors indicate optional blocks (e.g., KB, SpaCy POS/Dependency parsing).
- Fig. 2. STIX cyber graph example. Image credit: Introduction to STIX.

Note: Figures are referenced for context but not embedded. Please consult the original paper for visuals.

---

### References
1. Y. Zhang, Y. Wu, X. Li, and G. Zhao, “A systematic literature review on cyber threat intelligence sharing,” Information Systems Frontiers, vol. 10, no. 2, pp. 45–64, 2022.
2. R. Trifonov, S. Manolov, R. Yoshinov, G. Tsochev, and G. Pavlova, “Artificial intelligence methods for cyber threats intelligence,” International Journal of Computers, vol. 2, 2017.
3. A. Bécue, I. Praça, and J. Gama, “Artificial intelligence, cyber-threats and industry 4.0: Challenges and opportunities,” Artificial Intelligence Review, vol. 54, no. 5, pp. 3849–3886, 2021.
4. U. Zaratiana, N. Tomeh, P. Holat, and T. Charnois, “GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer,” arXiv:2311.08526, 2023.
5. S. Barnum, “Standardizing Cyber Threat Intelligence Information with the Structured Threat Information eXpression (STIX),” MITRE, pp. 1–22, 2012.
6. J. Piskorski and R. Yangarber, “Information extraction: Past, present and future,” ACM Computing Surveys, no. 3, pp. 213–239, 2013.
7. H. Gasmi, J. Laval, and A. Bouras, “Information Extraction of Cybersecurity Concepts: An LSTM Approach,” Applied Sciences, vol. 9, no. 19, p. 4137, 2019. Available: https://www.mdpi.com/2076-3417/9/19/4137
8. V. Legoy, M. Caselli, C. Seifert, and A. Peter, “Automated Retrieval of ATT&CK Tactics and Techniques for Cyber Threat Reports,” arXiv:2004.14322, 2020. Available: https://arxiv.org/abs/2004.14322
9. S. Weerawardhana, S. Mukherjee, I. Ray, and A. Howe, “Automated Extraction of Vulnerability Information for Home Computer Security,” in Foundations and Practice of Security, Springer, 2015, pp. 356–366.
10. T. Li, Y. Guo, and A. Ju, “A Self-Attention-Based Approach for Named Entity Recognition in Cybersecurity,” in 2019 15th Int’l Conf. on Computational Intelligence and Security (CIS), IEEE, 2019. Available: https://ieeexplore.ieee.org/document/9023758
11. P. Ranade, A. Piplai, A. Joshi, and T. Finin, “CyBERT: Contextualized Embeddings for the Cybersecurity Domain,” arXiv:2108.00382, 2021. Available: https://arxiv.org/pdf/2108.00382.pdf
12. N. Muennighoff, N. Tazi, L. Magne, and N. Reimers, “MTEB: Massive Text Embedding Benchmark,” arXiv:2210.07316, 2022. Available: https://arxiv.org/abs/2210.07316
13. X. Wang, R. Liu, J. Yang, R. Chen, Z. Ling, P. Yang, and K. Zhang, “Cyber Threat Intelligence Entity Extraction Based on Deep Learning and Field Knowledge Engineering,” Security & Communication Networks, 2023.
14. M. T. Alam, D. Bhusal, Y. Park, and N. Rastogi, “CyNER: A Python Library for Cybersecurity Named Entity Recognition,” arXiv:2204.05754, 2022.
15. Y. Zhou, Y. Tang, M. Yi, C. Xi, and H. Lu, “CTI View: APT Threat Intelligence Analysis System,” Security & Communication Networks, 2022.
16. Y. Zhou, Y. Ren, M. Yi, Y. Xiao, Z. Tan, N. Moustafa, and Z. Tian, “CDTier: A Chinese Dataset of Threat Intelligence Entity Relationships,” IEEE Transactions on Sustainable Computing, 2023.
17. F. Marchiori and M. Conti, “STIXnet: A Novel and Modular Solution for Extracting All STIX Objects in CTI Reports,” in ARES 2023, 2023. Available: https://arxiv.org/pdf/2303.09999.pdf
18. H. Touvron et al., “LLaMa: Open and Efficient Foundation Language Models,” arXiv:2302.13971, 2023.
19. X. Wang et al., “InstructUIE: An Instruction-tuned Universal Information Extraction Model,” arXiv:2301.13628, 2023.
20. C. Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,” arXiv:1910.10683, 2019.
21. H. W. Chung et al., “Scaling Instruction-Finetuned Language Models,” arXiv:2210.11416, 2022.
22. O. Sainz, O. L. de Lacalle, A. Soroa, and E. Agirre, “GoLLIE: Generalized Open-Domain Information Extraction,” arXiv:2304.08427, 2023.
23. B. Rozière et al., “CodeLLaMa: Open and Efficient Foundation Language Models for Code,” arXiv:2308.12950, 2023.
24. B. Zhou et al., “UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition,” arXiv:2303.08391, 2023.
25. S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, “C-Pack: Packaged Resources To Advance General Chinese Embedding,” 2023.
26. G. Rosa, L. Bonifacio, V. Jeronymo, H. Abonizio, M. Fadaee, R. Lotufo, and R. Nogueira, “In defense of cross-encoders for zero-shot retrieval,” arXiv:2212.06121, 2022.
27. A. Zubiaga, “Natural language processing in the era of large language models,” 2024.
28. H. Huang, Y. Qu, J. Liu, M. Yang, and T. Zhao, “An empirical study of LLM-as-a-judge for LLM evaluation: Fine-tuned judge models are task-specific classifiers,” arXiv:2403.02839, 2024.


## EXTENSION LINKS 
29. N. Wildt, D. M. Tartakovsky, S. Oladyshkin, and W. Nowak, “CODE: A global approach to ODE dynamics learning,” arXiv preprint arXiv:2511.15619, 2025. Available: https://arxiv.org/pdf/2511.15619
30. S. Oladyshkin and W. Nowak, “Data-driven uncertainty quantification using the arbitrary polynomial chaos expansion,” Reliability Engineering & System Safety, 106:179–190, 2012. Available: https://warwick.ac.uk/fac/sci/maths/research/grants/equip/grouplunch/2012OladyshkinNowak-ArbitraryPolynomialChaos.pdf

---

[^1]: The STIX taxonomy comprises 18 SDOs; however, certain ones are not pertinent to the NER task and instead serve as workaround instruments (e.g., Grouping SDO).
[^3]: Token Classification Models.
[^4]: `BAAI/bge-base-en-v1.5` backbone model (see Hugging Face).
[^5]: STIX relationship documentation: https://oasis-open.github.io/cti-documentation/stix/intro




