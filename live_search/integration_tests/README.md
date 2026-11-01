# Optional live search smoke test

For the separately opt-in ten-claim experiment, use
`test_live_evaluation.py --run-live --run-evaluation`. See the
[evaluation methodology and actual results](../docs/live_query_evaluation.md).
The ten-claim experiment archives responses and exports CSV/JSON comparisons;
selecting this directory without flags still performs no network requests.

These tests call the actual search clients against real provider endpoints.
They live outside `tests/`, so the normal mocked suite does not collect them.
Even explicit collection skips external calls unless `--run-live` is supplied.

```powershell
# Existing mocked suite (no network)
.\.venv\Scripts\python.exe -m pytest

# Confirm live tests remain opt-in
.\.venv\Scripts\python.exe -m pytest integration_tests -q

# Real APIs, one claim, up to five papers per provider
.\.venv\Scripts\python.exe -m pytest integration_tests --run-live -s -q -p no:cacheprovider
```

`-p no:cacheprovider` avoids pytest cache permissions affecting the live run.
The report is written even if a provider test fails, by default to
`artifacts/live-search-report.json`. This directory is ignored by Git.
Use `--live-report PATH`, `--live-claim "a scientific claim"`, and
`--live-limit 5` to customize the run. Limits must be between 1 and 100.

All three providers are tested regardless of production enable/disable flags:
the purpose is to check each adapter. Optional credentials are read from
`OPENALEX_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY`. Neither `.env` files nor
credentials from other stores are automatically loaded. No key is required
for arXiv. The test records authentication mode and checks the outgoing header
without printing keys. A successful keyed request shows the server accepted
that request; it does not independently prove that the server applied a
particular account's quota. Without keys, keyed authentication remains untested.

## Checks and report interpretation

- Real JSON/Atom response envelopes and parsed `ScientificPaper` objects.
- Raw versus parsed result counts, detecting records dropped during parsing.
- Per-field metadata availability: title, authors, DOI, date, year, landing URL,
  abstract, and PDF URLs. Counts and percentages use parsed papers as the
  denominator. Percentages are `null` when the result count is zero.
- `core_complete_count` means title, authors, publication date, DOI, and landing
  URL are all present. Missing optional fields do not fail a smoke test.
- HTTP statuses, selected quota headers, number of attempts, and actual waits.
  Retries use the production implementation, with a 15-second timeout per HTTP
  operation, two retries, and a 10-second maximum accepted retry delay.
- A valid empty result is `zero_results` and passes. HTTP/transport, auth, and
  rate-limit failures fail the corresponding test; their result count remains
  `null`, rather than falsely reporting zero papers. All providers still run.
- Rate limiting is exercised only if the service naturally returns 429.
  Successful calls report `not_exercised_no_429_observed`. The test does not
  deliberately flood a service. Deterministic rate-limit/Retry-After cases
  remain covered by the mocked suite.

Request hooks restrict every outgoing request to its provider's search
endpoint. Redirects remain disabled. No paper pages or PDFs are downloaded.
The report contains the submitted claim and safe diagnostics, not paper text,
credentials, or complete request URLs. Avoid sensitive claims when saving or
sharing reports. Reuse normal application adapters and coordinate external
quotas when running other clients at the same time; do not repeatedly rerun a
rate-limited test to try to bypass provider limits.

## Actual review run: 30 October 2026

Claim: `Exercise improves memory`; limit: five per provider; no API keys were
available in the environment. Network-enabled live tests produced:

| Provider | Outcome | Parsed/raw | Abstracts | PDF URLs | Core complete |
| --- | --- | --- | --- | --- | --- |
| OpenAlex | HTTP 200 | 5/5 | 4/5 | 2/5 | 4/5 |
| arXiv | HTTP 200 | 5/5 | 5/5 | 5/5 | 0/5 |
| Semantic Scholar | HTTP 429 on all three attempts | unavailable | unavailable | unavailable | unavailable |

OpenAlex and arXiv returned titles, authors, dates, years, and landing URLs for
all five records. DOI availability was 4/5 and 0/5 respectively, explaining
arXiv's core completeness count. Neither provider dropped records in parsing.
Semantic Scholar waited 1 and 2 seconds before exhausting its retry budget;
it supplied no Retry-After header. Its actual response parsing and keyed
authentication could not be verified in this run. OpenAlex and arXiv did not
return a rate-limit response, so their live backoff path was not exercised.

The mocked suite passed all 83 tests. Live outcome: two passed and one failed
due to Semantic Scholar's rate limit. The initial live invocation also emitted
pytest cache permission warnings; these did not affect HTTP requests or report
writing. No production defect was confirmed, and production code was unchanged.
Per-instance pacing and the lack of shared pacing across one-shot pipeline
runs remain the existing documented limitations; this smoke test does not
claim to validate distributed quota coordination.
