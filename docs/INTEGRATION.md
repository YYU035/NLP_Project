# Integration contract

Flow: claim → live search → normalized papers → BM25 and SBERT → RRF → optional Cross-Encoder → PDF/abstract evidence → DeBERTa-MNLI → aggregate verdict → grounded explanation → Streamlit.

Use `shared/contracts.py` between modules. Paper IDs must be stable across search providers and SciFact. Deduplicate using normalized DOI, arXiv ID, then conservative title matching; retain provider identifiers and provenance. Publication dates may be unknown and date filtering must define its missing-date policy.

Retrieval produces separate BM25 and SBERT ranked lists. Fusion consumes rank positions, not incomparable raw scores. Preserve method names and stage timings. Evidence must identify a paper and a text location; abstract fallback must be visible. Preserve SciFact sentence indices for rationale evaluation.

Map model-specific NLI labels explicitly into entailment, contradiction and neutral. Document aggregation thresholds and mixed-evidence handling. A Supported or Contradicted verdict needs source evidence. Handle no usable evidence as Insufficient Evidence.

Explanations must use the returned evidence and verdict. Cite evidence IDs and paper metadata. Never invent sources or let the LLM override the verification verdict. Optional corrections must be evidence-grounded.

`shared/pipeline.py` is the future integration entry point. Benchmark execution uses the fixed SciFact corpus, not live search results. Live demos exercise the same downstream stages with API papers and report partial failures.
