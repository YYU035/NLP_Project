# Scientific paper retrieval — NTU NLP RAG, Person 1

Python 3.11+ module supplying scientific paper candidates to Person 2's
BM25/SBERT ranking stage. It returns normalized Pydantic metadata, deduplicated
across providers, with the original claim and complete retrieval provenance.

| Provider | Query strategy | Depth |
|---|---|---|
| OpenAlex | Local KeyBERT keyphrases | 30 per claim |
| OpenAIRE | Local KeyBERT keyphrases | 30 per claim |
| arXiv | spaCy scientific concepts | 30 per claim |

arXiv relaxation is **disabled by default**. `top_k` is per provider: the
unranked union may contain up to 90 papers at the default depth. Ranking, PDF
downloads, full-text verification and claim verdicts are outside this module.
Paper relevance labels in the research artifacts are provisional judgments.

## Install

From a clone of this repository, Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,keybert,nlp-model]"
# Explicit setup download of the existing pinned local model; no paper searches.
.venv\Scripts\python.exe -m paper_search.model_setup
```

Use any installed Python 3.11+ interpreter to create the environment. On
macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` instead.
Install the `keybert` and `nlp-model` extras for the finalized retrieval policy;
`dev` provides pytest/Ruff. The model setup command requires internet access.

KeyBERT uses CPU-local `sentence-transformers/all-MiniLM-L6-v2`, pinned to
revision `2a7cca6ad7605f3183bce07f677bda658059bbe9`; spaCy uses
`en_core_web_sm` 3.8.0. Weights live in ignored
`artifacts/local-models/all-MiniLM-L6-v2/` and are never downloaded implicitly
during retrieval. There are no hosted LLM calls in the retrieval path.
Dependency versions from the experiments are recorded in their frozen plans.

The default model path is anchored to this checkout for editable installs, so
Person 2 can call retrieval from a different working directory. For other
installation layouts, inject `KeyBERTQueryGenerator(absolute_model_path)` into
`RetrievalService(generator=...)`; see the [handoff guide](docs/retrieval_integration.md).

## Configuration

`.env.example` contains blank credentials and the finalized provider flags.
Copying it to `.env` **does not load it**. Export variables in your shell or
configure your IDE to load them. Do not commit real credentials.

| Variable | New retrieval-service behavior |
|---|---|
| `PAPER_SEARCH_OPENALEX_ENABLED` | Enabled unless explicitly disabled |
| `PAPER_SEARCH_OPENAIRE_ENABLED` | Enabled unless explicitly disabled |
| `PAPER_SEARCH_ARXIV_ENABLED` | Enabled unless explicitly disabled |
| `PAPER_SEARCH_TIMEOUT_SECONDS` | 30 seconds per HTTP operation |
| `OPENALEX_API_KEY` | Optional provider key; masked in settings |
| `OPENAIRE_TOKEN` | Optional provider token; masked in settings |

Flags accept `true`/`false` or `1`/`0`. Public access is subject to provider
availability and quotas. Reuse a service across claims to preserve pacing;
coordinate quotas across processes. The shared transport retains bounded
retries, timeouts, `Retry-After` handling and provider failure isolation.

`retrieve(..., top_k=30)` controls depth explicitly; it does not read the legacy
`PAPER_SEARCH_MAX_RESULTS`. `MAX_QUERIES`, `QUERY_MODE`, `QUERY_MODEL` and
Semantic Scholar settings remain for historical interfaces/experiments.
Semantic Scholar and the separate Gemini evaluator are **not used** by the
finalized retrieval service. Explicit `Settings()` objects keep their legacy
defaults; set `openaire_enabled=True` when supplying one to the service.

## Person 2 integration

```python
import asyncio
from paper_search.retrieval import retrieve

async def main() -> None:
    result = await retrieve("Physical exercise improves memory", top_k=30)
    for entry in result.papers:
        paper = entry.paper
        # Person 2 ranks title/abstract against result.original_claim.
        print(entry.identity, paper.title, paper.abstract)
        print(paper.doi, paper.landing_url, paper.pdf_urls)
        print(entry.provenance)
    print({name: run.status for name, run in result.providers.items()})

asyncio.run(main())
```

The [minimal working example](examples/person2_integration.py) prints the full
JSON handoff, including identifiers, metadata, provenance and provider errors:

```powershell
# Performs live retrieval when executed; not an evaluation experiment.
.venv\Scripts\python.exe examples/person2_integration.py "Physical exercise improves memory"
```

For multiple claims, use one persistent context:

```python
from paper_search.retrieval import RetrievalService, RetrievalUnavailable

async def retrieve_batch(claims: list[str]):
    async with RetrievalService() as service:
        for claim in claims:
            try:
                result = await service.retrieve(claim, top_k=30)
            except RetrievalUnavailable as exc:
                result = exc.result  # Diagnostics for a complete outage.
            yield result
```

The service returns `RetrievalResult`, with `original_claim`, `papers`,
`providers`, elapsed seconds and actual HTTP attempts. Each candidate has
`paper: ScientificPaper`, canonical identity/aliases, query/provider/rank
provenance and original normalized source records. Titles, abstracts, DOI,
dates, authors and URLs may be missing; preserve `None`/empty values.
OpenAIRE descriptions are not guaranteed abstracts. PDF URLs are metadata only.

Valid zero results differ from provider failures. Partial failures retain
available candidates; complete failure raises `RetrievalUnavailable` carrying
the diagnostic result. Invalid inputs fail validation. Deduplication uses
DOI, version-normalized arXiv IDs and conservative exact-title matching with
year/author checks; conflicting identifiers are not merged. There is no fuzzy
matching or global ranking. Keep identity aliases and provenance when ranking.

See [the complete integration contract](docs/retrieval_integration.md).

## Offline quality checks

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m pytest integration_tests
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pip check
```

Unit tests use mocked HTTP/providers and local NLP fixtures. Integration tests
are skipped without their explicit live flags; the command above makes no API
requests. Some tests use preserved artifacts, so retain the research folders.
Live-test instructions remain in [integration_tests/README.md](integration_tests/README.md).

## Repository layout and research evidence

```text
src/paper_search/       Installable package; production has no script imports
  retrieval.py         Public retrieve / persistent RetrievalService
  retrieval_models.py  Person 2 handoff and provider diagnostics
  clients/             Provider parsing, authentication, retries and pacing
  keybert_queries.py   Existing local embedding query generation
  linguistic.py        Existing spaCy extraction
  query_rendering.py   Provider syntax
  arxiv_rendering.py   arXiv scientific concept selection
  deduplication.py     Conservative cross-provider identities
  arxiv_fallback.py    Opt-in bounded relaxation; default off
  model_setup.py       Explicit pinned model setup only
  evaluation/          Research evaluation utilities
  relevance/           Separate optional relevance evaluator
examples/              Minimal team integration
tests/                 Offline unit/regression tests
integration_tests/     Explicit opt-in live tests
scripts/               Historical experiments, audits and offline analysis
docs/                  Design, integration and submission review
artifacts/             Preserved plans, reports, labels, CSVs and responses
```

Legacy `SearchPipeline`, Semantic Scholar, deterministic ablation modes and the
PDF-verification placeholder are retained for compatibility/reproducibility.
They do not select the finalized policy. `SearchPipeline` concatenates results;
use `RetrievalService` for the deduplicated handoff.

- [Research artifact index](artifacts/README.md)
- [Experiment script index](scripts/README.md)
- [Final depth/fallback results](artifacts/readiness-live-results-20261101/report.md)
- [Label reconciliation and strategy comparison](artifacts/keybert-experiment-20261101/final_strategy_comparison.md)
- [Submission-readiness review](docs/submission_readiness.md)

Preserved reports describe their historical run, not current defaults. Do not
rerun completed experiments just to set up or integrate the module. Current
evidence supports a candidate-generation choice, not improved claim accuracy;
the fallback additions and many deeper candidates still need assessment.

Git ignores local model weights/caches, virtual environments, credentials,
runtime databases and `outputs/`; it retains essential research evidence under
`artifacts/`. Do not use force-add to include secrets or weights. This workspace
has no Git metadata, so tracking/history must be checked in the actual clone
before pushing. No commit or push is performed by setup or retrieval.
