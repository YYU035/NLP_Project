# Research Claim Verification

An evidence-based system for checking scientific claims against research papers. The project combines live paper search, lexical and semantic retrieval, hybrid ranking, natural language inference, and grounded explanations in a Streamlit interface.

Given a claim, the system will find relevant papers, identify evidence passages, and return one of three verdicts: **Supported**, **Contradicted**, or **Insufficient Evidence**, together with citations and an explanation.

> **Project status:** Planning and initial folder setup. The modules described below are planned work and have not been implemented yet.

## System Workflow

```text
User Claim
    |
    v
Live Research Paper Search
(arXiv / Semantic Scholar / OpenAlex)
    |
    v
Publication Date Filtering & Deduplication
    |
    v
BM25 Retrieval + SBERT Retrieval
    |
    v
Reciprocal Rank Fusion (RRF)
    |
    v
Optional Cross-Encoder Reranking
    |
    v
PDF Text Extraction & Chunking
(Abstract fallback when full text is unavailable)
    |
    v
Relevant Evidence Passages with Source References
    |
    v
DeBERTa-MNLI Claim–Evidence Predictions
    |
    v
Verdict Aggregation
(Supported / Contradicted / Insufficient Evidence)
    |
    v
Evidence-Grounded Explanation & Streamlit Display
```

## Modules and Team Responsibilities

### Person 1 — Live Research Paper Search · Raghavi

**Folder:** `live_search/`

- Integrate arXiv, Semantic Scholar, or OpenAlex APIs.
- Search for relevant papers based on user claims.
- Filter results by publication date and remove duplicate papers.
- Return paper metadata, including titles, authors, publication dates, abstracts, and source links where available.

### Person 2 — Retrieval · Chong Yao

**Folder:** `retrieval/`

- Implement BM25 for lexical retrieval.
- Implement SBERT for semantic retrieval.
- Rank candidate papers by relevance to the claim.
- Compare lexical and semantic retrieval performance.

### Person 3 — Hybrid Ranking & Evaluation · Yu Yue

**Folder:** `ranking_evaluation/`

- Combine BM25 and SBERT rankings using Reciprocal Rank Fusion (RRF).
- Implement optional Cross-Encoder reranking.
- Evaluate retrieval using Recall@K, MRR, nDCG, and latency.
- Conduct comparative experiments and ablation studies.

### Person 4 — Evidence Extraction & Claim Verification · Chong Yao

**Folder:** `evidence_verification/`

- Extract and chunk available PDF full text, with abstract fallback when necessary.
- Identify relevant evidence passages and retain source references.
- Apply pretrained DeBERTa-MNLI to claim–evidence pairs.
- Aggregate predictions into Supported, Contradicted, or Insufficient Evidence.
- Evaluate claim verification using Accuracy and Macro-F1 on labeled data.

### Person 5 — Grounded Explanation & Frontend · Kireeti

**Folder:** `explanation_frontend/`

- Use an LLM to generate explanations based on retrieved evidence and verification results.
- Generate evidence-grounded explanations and optional claim corrections.
- Display verdicts, evidence passages, citations, and relevant papers.
- Build the Streamlit interface.

### Shared — Dataset & Integration · Everyone

**Folder:** `shared/`

- Use SciFact as a fixed benchmark for reproducible evaluation.
- Agree on shared data formats and interfaces between modules.
- Integrate the five modules into one end-to-end pipeline.
- Test the complete system using live research paper search.

## Project Structure

```text
NLP_Project/
├── live_search/              # Live paper search, filtering, and deduplication
├── retrieval/                # BM25 and SBERT retrieval
├── ranking_evaluation/       # RRF, reranking, experiments, and retrieval metrics
├── evidence_verification/    # Evidence extraction, NLI, and verdict aggregation
├── explanation_frontend/     # Grounded explanations and Streamlit interface
├── shared/                   # Dataset preparation and shared integration work
├── .gitignore
└── README.md
```

Each contributor owns the implementation and internal structure of their assigned module. The folders currently contain only `.gitkeep` files so Git can track them.

## Evaluation Plan

### Fixed Benchmark: SciFact

Use a fixed SciFact corpus and labeled claims to compare methods under reproducible conditions. Record the dataset version, splits, model versions, and experiment settings. Keep benchmark evaluation separate from live API search, whose results can change over time.

### Retrieval Evaluation

| Metric | Purpose |
| --- | --- |
| Recall@K | Measure how many relevant papers appear among the top K results |
| MRR | Measure the rank of the first relevant paper |
| nDCG@K | Measure ranking quality with higher weight on relevant papers near the top |
| Latency | Measure retrieval, reranking, and end-to-end response time |

Planned comparisons:

- BM25 alone.
- SBERT alone.
- BM25 + SBERT combined with RRF.
- RRF with Cross-Encoder reranking.

Ablation studies will examine the contribution of each retrieval branch and optional reranking stage.

### Claim Verification Evaluation

Report **Accuracy** and **Macro-F1** for the three verdict classes. Document how SciFact annotations map to system verdicts and distinguish evaluation with gold evidence from evaluation with retrieved evidence.

### Live Integration Testing

Run user claims through live paper search and the complete downstream pipeline. Check that paper metadata, evidence references, verdicts, and explanations remain connected throughout the process, including when PDF full text is unavailable.

## Planned User Output

For each claim, the interface will show:

- The claim and final verdict.
- Relevant evidence passages with source references.
- Research paper metadata and citations.
- An explanation grounded in the displayed evidence.
- An optional evidence-grounded claim correction, when appropriate.

Unavailable or insufficient evidence should be made explicit. Explanations must not invent evidence or citations.

## Setup and Usage

Installation requirements and run commands will be added as the modules are implemented and integrated. There is currently no runnable application or benchmark command.
