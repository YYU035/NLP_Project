# Scientific paper retrieval

The retrieval module finds scientific papers and passes
normalized candidates to downstream ranking. It returns the original claim,
paper metadata and retrieval provenance.

## Architecture

| Provider | Query strategy |
|---|---|
| OpenAlex | KeyBERT |
| OpenAIRE | KeyBERT |
| arXiv | spaCy concepts |

Each provider returns up to 30 papers per claim. Results are deduplicated using
DOI, arXiv IDs and conservative title matching, giving at most 90 candidates
before downstream ranking. arXiv query relaxation is disabled by default.
Ranking, PDF downloads and claim verification are outside this module.

## Installation

Run these commands from the module directory (the folder containing this
README and `pyproject.toml`; `live_search/` in the group repository):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,keybert,nlp-model]"
.venv\Scripts\python.exe -m paper_search.model_setup
```

Python 3.11+ is required. On macOS/Linux, use `python3` to create the environment
and `.venv/bin/python` for the remaining commands.

Setup downloads the existing MiniLM model for local CPU inference, pinned to
revision `2a7cca6ad7605f3183bce07f677bda658059bbe9`. spaCy uses
`en_core_web_sm` 3.8.0. Weights are not committed or downloaded during retrieval.

Use [.env.example](.env.example) for configuration. Export variables in your
shell or load them through your IDE; copying the file to `.env` does not load
it automatically. OpenAlex and OpenAIRE accept optional credentials. Keep real
keys out of Git.

## Usage

```python
import asyncio
from paper_search.retrieval import retrieve

result = asyncio.run(retrieve("Physical exercise improves memory", top_k=30))
for entry in result.papers:
    print(entry.paper.title, entry.paper.abstract)
    print(entry.paper.doi, entry.paper.landing_url, entry.paper.pdf_url)
    print(entry.provenance)
```

This makes live API requests. `top_k` is a per-provider limit, not a global
ranking. Titles, abstracts and URLs may be missing. PDF URLs are metadata;
their accessibility and contents have not been verified.

Results retain available papers if one provider fails. A complete outage raises
`RetrievalUnavailable`, which contains provider diagnostics. For batches, reuse
one `RetrievalService` context to preserve pacing and query caching. See the
[integration guide](docs/retrieval_integration.md) and
[working example](examples/person2_integration.py).

## Testing

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

The normal suite uses mocked requests and local fixtures. Real API checks are
separate and require explicit opt-in; see [integration tests](integration_tests/README.md).

## Next steps

Connect the candidate output to downstream BM25/SBERT ranking and assess the
remaining relevance labels before deciding whether to enable arXiv relaxation.
The [experiment summary](artifacts/README.md) explains the current choices.
[Research scripts](scripts/README.md) support reproducing the comparisons;
they are not required for normal retrieval.
