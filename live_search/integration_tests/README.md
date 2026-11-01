# Integration tests

These tests check real API responses, metadata parsing, authentication and
provider errors. They are separate from the normal offline test suite and
skip live requests unless `--run-live` is supplied.

Run commands from the module directory after installing the dependencies:

```powershell
# Offline unit tests
.venv\Scripts\python.exe -m pytest

# Collect integration tests without making API calls
.venv\Scripts\python.exe -m pytest integration_tests -q

# Live smoke test: one claim, up to five papers per provider
.venv\Scripts\python.exe -m pytest integration_tests/test_live_search.py --run-live -s -q
```

The smoke test covers OpenAlex, arXiv and Semantic Scholar adapters, regardless
of the enabled-provider flags. It does not test OpenAIRE or the full current
routing policy. Optional keys are read from `OPENALEX_API_KEY` and
`SEMANTIC_SCHOLAR_API_KEY`; `.env` files are not loaded automatically.

Use `--live-claim "your claim"`, `--live-limit 5` and `--live-report PATH` to
change the smoke test inputs. Its default report is
`artifacts/live-search-report.json`, which is ignored by Git.

A successful empty response is recorded separately from an API failure. Reports
include result counts, metadata availability, request attempts and observed
rate limits. No paper pages or PDFs are downloaded. Live rate-limit handling is
only observed when an API actually returns 429; repeatable retry cases belong
to the mocked tests.

The earlier ten-claim evaluation requires both flags:

```powershell
.venv\Scripts\python.exe -m pytest integration_tests/test_live_evaluation.py --run-live --run-evaluation --evaluation-dir outputs/live-evaluation-new -s -q
```

This makes multiple live requests. Use a fresh output path and check the
[evaluation protocol](../docs/live_query_evaluation.md) first. For the finalized
routing and depth comparisons, read the [saved results](../artifacts/README.md).
