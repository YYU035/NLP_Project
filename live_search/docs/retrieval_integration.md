# Person 2 retrieval contract

Use the additive service for the agreed policy: KeyBERT on OpenAlex/OpenAIRE and spaCy on arXiv. The legacy `SearchPipeline` is unchanged. Existing optional KeyBERT dependencies, pinned local `artifacts/local-models/all-MiniLM-L6-v2` weights and spaCy model must already be installed. No model is downloaded implicitly. Load environment variables before creating the service (`Settings.from_environment()` does not load a `.env` file itself).

Install with `pip install -e ".[keybert,nlp-model]"` and explicitly download the
existing weights with `python -m paper_search.model_setup`. No experimental
script is required. The editable-install default model path is anchored to
the checkout; changing the caller's working directory does not change it.
Other installations can pass an existing generator explicitly:

```python
from pathlib import Path
from paper_search.keybert_queries import KeyBERTQueryGenerator
from paper_search.retrieval import RetrievalService

# Replace the path with your actual local model directory.
service = RetrievalService(
    generator=KeyBERTQueryGenerator(Path("/your/local/all-MiniLM-L6-v2"))
)
```

The [minimal integration example](../examples/person2_integration.py) uses only
the installed package and prints the full handoff JSON. Executing it performs
live retrieval; importing it does not.

```python
from paper_search.retrieval import RetrievalService, RetrievalUnavailable

async def discover(claim: str):
    async with RetrievalService() as service:
        try:
            result = await service.retrieve(claim, top_k=30)
        except RetrievalUnavailable as exc:
            # All providers failed; diagnostics remain available.
            return exc.result
        return result
```

For a batch, put the context **outside** the claim loop and reuse it to preserve provider pacing/hourly budgets and the local query cache. Flags/keys use existing environment configuration. This new interface enables OpenAIRE by default unless explicitly disabled; Semantic Scholar is outside the agreed policy. Explicit `Settings` objects retain their flags (set `openaire_enabled=True` when desired). Never create multiple service instances to bypass rate limits.

`result.papers` is an **unranked, deduplicated union**. `top_k=30` requests at most 30 candidates per provider, potentially 90 distinct candidates overall. It is not the global ranked top 30. Person 2 should rank `entry.paper` using `result.original_claim` and retain `entry.identity`, `identity_aliases` and `provenance`. All original normalized records remain in `entry.source_records`. Preferred identity uses DOI, then arXiv ID, then provider ID; new metadata can change the preferred identifier, so retain aliases when reconciling later results. Conservative missing-ID/title rules intentionally leave some duplicates unresolved.

`result.providers[name]` distinguishes `success`, `zero_results`, `provider_failure`, `query_generation_failure` and `partial_failure`. It contains provider elapsed seconds and query records; each query includes client input, stage, strategy, parsed count, errors, elapsed seconds and actual HTTP attempts. Attempts contain endpoint and safe query parameters (the exact arXiv expression including dates), status and duration; authentication headers are excluded. Provider-call latency includes pacing/retries. Full-run latency includes generation and concurrent provider work; it is not the sum of provider latency. `result.model_dump_json()` serializes the handoff, including total HTTP attempts.

The original claim is preserved exactly in the handoff. Queries are discovery reformulations, not replacements for the claim during ranking or evidence assessment. spaCy extraction omissions are recorded. Results may lack titles, abstracts, dates, authors or PDF URLs. OpenAIRE descriptions are not necessarily scientific abstracts. PDF links are not verified full text.

Opt-in experimentation: `RetrievalService(arxiv_fallback=True, max_fallbacks=2)` enables bounded fallback below `min(5, top_k)` unique arXiv candidates. By default all selected concepts are required, so only phrase-order relaxation is possible. A custom `planner` can mark reviewed optional phrases via `ProviderPlan.optional_phrases`; its required+optional phrases must exactly render the primary query. Required phrases must include every central concept/outcome that must be retained. Fallback unions preserve primary candidates first. No fallback follows a provider failure. It remains off by default pending evaluation.

Optional `archive=Path("fresh-run")` saves response bodies with SHA-256 and redacts configured authentication values. Use a fresh archive directory per service; files cannot be overwritten.

The one-shot `paper_search.retrieval.retrieve(claim, top_k=30)` convenience function is available for single requests; use the persistent service for repeated calls. The frozen experiment runner has a separate deliberately zero-retry budget and does not alter production retries.

See the [readiness report](../artifacts/integration-readiness-20261101/report.md) for the completed baseline audit, blinded manual sample, tests and pending live experiments. Integration is tested offline; no new service live validation has been run.
