# Research Claim Verification

A fresh project skeleton for live paper search, retrieval, hybrid ranking, evidence verification, and grounded explanations.

## Ownership

| Module | Owner | Scope |
| --- | --- | --- |
| `live_search/` | Raghavi | arXiv / Semantic Scholar / OpenAlex, claim queries, publication date filters, deduplication, metadata |
| `retrieval/` | Chong Yao | BM25, SBERT, lexical vs semantic retrieval |
| `ranking_evaluation/` | Yu Yue | RRF, optional Cross-Encoder, Recall@K / MRR / nDCG / latency, comparisons and ablations |
| `evidence_verification/` | Chong Yao | PDF extraction and chunking, abstract fallback, cited evidence, DeBERTa-MNLI, verdict aggregation, Accuracy / Macro-F1 |
| `explanation_frontend/` | Kireeti | Evidence-grounded LLM explanations, optional corrections, Streamlit interface |
| `shared/` | Everyone | Data contracts, SciFact adapter, end-to-end integration |

## Layout

```text
live_search/              # providers/, search orchestration and normalization
retrieval/                # BM25 and SBERT implementations
ranking_evaluation/       # fusion, reranking, metrics and experiments
evidence_verification/   # PDF, evidence and NLI pipeline
explanation_frontend/     # explanations, prompts and Streamlit
shared/                   # contracts, datasets and pipeline
configs/                  # reproducible run configuration
data/                    # SciFact, downloaded papers and processed artifacts
scripts/                  # dataset preparation and experiment entry points
tests/                    # unit, integration and benchmark checks
docs/                     # integration and evaluation specifications
```

The modules are intentionally unimplemented. Only shared data contracts and module boundaries exist; there are no working API integrations, models, benchmark results, or frontend yet.

## Development

Use Python 3.10+ and install the package with `python -m pip install -e .`.
Add runtime dependencies when implementing each module, then record exact versions for reproducible experiments. Never commit credentials, downloaded PDFs, model weights, or benchmark outputs.

Read [integration](docs/INTEGRATION.md) and [evaluation](docs/EVALUATION.md) before implementing a module. Keep SciFact benchmark evaluation separate from live API demos.
