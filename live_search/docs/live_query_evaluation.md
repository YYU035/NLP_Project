# Live API validation and preliminary query reformulation evaluation

The experiment uses the existing clients and query generator. It performs
read-only search requests; it does not run LLMs, BM25/SBERT ranking, PDF
downloads, or claim verification. Results measure retrieval coverage, not
relevance, truth, or accuracy.

## Actual run: 30 October 2026

Ten diverse claims were tested against all three providers, using five results
per query and the inclusive date window 2020-01-01 to 2026-10-30. No API keys
were configured. Seven claims generated an alternative; three retained only
their original query. Provider request pacing and the existing two-retry
policy were retained, with an explicit 15-second timeout per HTTP operation.

| Provider | Original API outcomes | Original unique papers | Original + concepts unique papers | Additional identities |
| --- | --- | ---: | ---: | ---: |
| OpenAlex | 10 valid nonempty responses | 50 | 63 | 13 |
| arXiv | 1 nonempty, 9 valid empty responses | 5 | 5 | 0 |
| Semantic Scholar | 1 valid empty response, 9 rate-limited failures | incomplete | incomplete | not comparable |

All 24 OpenAlex calls and all 24 arXiv calls succeeded, including matched-budget
controls. Semantic Scholar had 3 valid outcomes and 21 rate-limited failures
across 24 calls. Its three valid responses were one original zero-result
response, one repeated-original zero-result response, and one repeated-original
response containing five real papers. That five-paper control does not replace
failed observations in the natural mode comparison.

There were **72 logical client calls and 116 actual HTTP attempts**. The 116
raw response bodies and their recorded SHA-256 hashes were verified locally.
Semantic Scholar produced 65 HTTP 429 responses; all attempts remained bounded.
One successful control recovered after a 429, and a valid empty original also
recovered after a 429. Keyed authentication was not tested; outgoing anonymous
authentication-header checks passed.

The original-only mode cost 49 HTTP attempts across providers. Adding the 21
logical concept queries cost 35 additional attempts, for 84 natural-mode
attempts. Fresh repeated-original controls cost another 32 attempts as
experimental overhead, producing the physical total of 116. These totals count
the common original request once, even though both modes reuse its response.

Observed call-duration sums, including pacing and retries, were 88.25 seconds
for original calls, 64.58 additional seconds for concept calls, and 53.22 seconds
for repeat controls. Providers ran concurrently; the full live request phase
took about 111 seconds. These are observed scheduled durations, not independent
end-to-end A/B latency estimates.

## Validation and metadata

Original OpenAlex calls returned 50 records: all had titles, authors, publication
dates, and landing URLs; 49 had DOIs, 46 had abstracts, and 45 had PDF URLs.
The five arXiv records all had titles, authors, dates, abstracts, landing URLs,
and PDF links; one had a DOI. The successful Semantic Scholar repeat control
had five titles/authors/dates/DOIs, four abstracts, and three PDF links.

No parsed date was outside the requested window. This verifies the returned
sample, not the completeness of any provider's date index. arXiv dates refer
to original submissions. Missing dates would remain unknown, not count as
filter violations. PDF links are available metadata only; no links were fetched.

The live run exposed a narrow parser defect: Semantic Scholar returned
HTTP 200 with `{"total": 0, "offset": 0}` and omitted the `data` array twice.
The client previously rejected this explicit empty response. A regression
fixture and strict special-case fix now accept it; nonzero totals, arbitrary
missing arrays, null arrays, and Boolean totals remain errors. The exact two
saved bodies were replayed through the corrected production client using
`httpx.MockTransport`, without new network requests. The original statuses and
errors remain in `calls.jsonl`, with corrections documented in `results.json`.

## Coverage and cost examples

- **Observed coverage gain:** OpenAlex battery engineering (`c02`) increased
  from 5 to 8 unique identities. Power electronics (`c03`) also increased from
  5 to 8. Relevance of the extra papers has not been assessed.
- **No difference:** OpenAlex astronomy (`c05`), qualified nutrition (`c07`),
  and the short memory claim (`c10`) generated no alternate query and retained
  the same coverage without extra query cost.
- **Apparent harm in cost:** arXiv's seven alternate queries returned valid
  empty responses, added no identities, and consumed seven extra requests
  plus pacing time. This is extra cost without observed coverage benefit,
  not evidence of worse scientific accuracy.
- **Matched coverage harm:** none was observed in complete comparisons. This
  does not establish that reformulation cannot harm retrieval elsewhere.

## Natural and matched budgets

The original response is fetched once and reused by both modes. Natural
comparison is one original query versus the union of original and concept
query responses. This union cannot lose an original identity by construction;
that containment must not be mistaken for improved recall.

For each of the seven claims with alternatives, the matched logical budget
compares **original + fresh original repeat** with **original + concept query**.
Claims without alternatives use one query in both arms. Extra-call order is
counterbalanced by claim index. Retries can make physical budgets unequal;
JSON/CSV explicitly report logical and actual HTTP budgets and mismatch flags.

OpenAlex and arXiv had ten complete matched pairs each, including three
one-request pairs. Their matched baselines remained 50 and 5 identities,
respectively; reformulation unions contained 63 and 5. No complete Semantic
Scholar matched pair was available. An identical-query repeat can be served
by upstream caches and is a weak two-query baseline, not an optimized competing
query strategy. This preliminary control should not support broad conclusions.

No reliable relevance labels exist for these claims. Precision@K and Recall@K
are therefore `null`, not estimated from counts, titles, or provider scores.
Only unique returned identities, overlap/Jaccard, additions/losses, successful-
outcome zero rates, request counts, and observed latencies are reported.

## Reproduce or inspect the run

The completed artifacts are in `artifacts/query-evaluation-20261030/`:

| File | Contents |
| --- | --- |
| `report.md` | Actual per-provider/per-claim comparison and limitations |
| `results.json` | Complete analysis, paper groups, queries, IDs, corrections |
| `run.json` | Initial claims, filters, generator diagnostics, safe configuration |
| `calls.jsonl` | Immutable checkpointed original runtime observations |
| `calls.csv` | Query outcomes, requests, timings, metadata, errors |
| `papers.csv` | Every normalized paper and its provider ID/DOI/metadata |
| `comparisons.csv` | Natural/matched per-provider and per-claim metrics |
| `claim_comparisons.csv` | Cross-provider observed coverage with partial flags |
| `provider_summary.csv` | Aggregate coverage, failures, counts, denominators |
| `responses/*.body` | Every response body, including error responses |

Raw response headers, exact query parameters, timestamps, timings, and hashes
are attached to attempts in JSON. Auth header values are never recorded;
reflected credentials would be redacted and flagged. Bodies are saved after
httpx content decoding, not as compressed transport bytes. Search-only endpoint
allowlists prevent any PDF or landing-page request. Use a fresh output directory
for a new experiment; existing evidence is not overwritten.

```powershell
# New live run, honoring enabled providers and keys from Settings.
.\.venv\Scripts\python.exe scripts/run_query_evaluation.py --run-live --from-date 2020-01-01 --to-date 2026-10-30 --limit 5 --timeout 15

# Regenerate analysis and replay supported sparse-empty corrections offline.
.\.venv\Scripts\python.exe scripts/run_query_evaluation.py --summarize-existing artifacts/query-evaluation-20261030

# Offline unit suite and lint; these do not contact APIs.
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

The CLI requires `--run-live` for external requests and exits nonzero after
saving results if any provider failed. The actual run was degraded; it was
not reported as a fully passing live test. To execute the same experiment as
explicit live pytest integration checks, separately from the mocked suite:

```powershell
.\.venv\Scripts\python.exe -m pytest integration_tests/test_live_evaluation.py --run-live --run-evaluation --evaluation-dir artifacts/new-live-evaluation -s
```

Selecting `integration_tests/` alone skips all external calls. The new ten-claim
test additionally requires `--run-evaluation`, so the existing one-claim smoke
test does not unexpectedly trigger a larger experiment. This pytest command
would perform a new live run; it was not redundantly executed after the CLI
experiment and Semantic Scholar's rate limits.

## Limitations

The claim sample is small and hand-selected, many claims are long and highly
conjunctive, and each response is truncated to five papers. Default provider
ordering, updates, caches, tokenization, rate limits, and anonymous quotas vary.
Query broadening can move relevant papers out of a fixed top-k list. The
absence of observed new papers is not evidence against the scientific claim.

Deduplication uses normalized DOI, version-independent arXiv IDs, and exact
NFC/whitespace-normalized long titles with the same known year and a shared
full author. Scientific title case/symbols are retained. Conflicting known IDs
never merge, including through an identifier-less title bridge. No fuzzy title
matching is used. Missing IDs, different titles, or author variations can leave
duplicates and inflate observed unique/additional counts. Relevance and true
work identity require further manual evaluation.

Most Semantic Scholar observations are unavailable. Reporting its successful-
original zero rate as 1/1 is not a result for the other nine failed claims.
No inference about that provider's relative accuracy is justified. Matched
wire-request counts may differ even when logical budgets match. No statistical
significance or generalized retrieval improvement is claimed.
