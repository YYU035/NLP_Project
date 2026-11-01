# arXiv concept discovery and controlled evaluation

Implemented on 31 October 2026. The shared `QueryGenerator.generate()` contract
and `original_only` / `original_plus_concepts` modes are unchanged. Originals
retain their exact input string; generated queries contain no provider syntax.
OpenAlex and Semantic Scholar production rendering are unchanged.

## Backend policy

[arxiv_rendering.py](../src/paper_search/arxiv_rendering.py) calls the existing
spaCy concept extractor and conservative concept selector. It takes up to two
substantive concepts in source order, excluding reporting phrases and numerical
conditions. A small explicit rule can retain measurement heads such as
`capacity` from `90% capacity`, while recording the removed quantity. Scientific
names starting with numbers, such as `2D materials`, remain eligible. Generic
measurement wrappers such as `expression of` can be removed without changing
the spelling of their referent.

[query_rendering.py](../src/paper_search/query_rendering.py) renders these as
arXiv `all:` constraints. Multiword concepts and syntax-sensitive terms receive
quotes; ordinary single words do not. For example:

```text
all:"graph neural networks" AND all:"molecular property prediction"
all:"lithium-ion batteries" AND all:capacity
all:"TNF-α" AND all:"COVID-19"
```

This uses documented [arXiv field/phrase/Boolean syntax](https://info.arxiv.org/help/api/user-manual.html).
There are no synonyms, stemming, LLM calls or attempts to interpret evidence.
The renderer returns `original`, `selected_concepts`, `omitted_spans`, extraction
status and warnings. These diagnostics are saved for every experimental claim.
Omitted spans refer to whitespace-normalized rendering input; `original` retains
the exact input. An extraction failure uses the original as one quoted phrase
and warns, rather than sending an empty query or every-word conjunction.

[ArxivClient.search()](../src/paper_search/clients/arxiv.py) now uses this policy
before adding the existing submission-date filter. Its search signature is
unchanged. `client.query_strategy = "literal"` retains the legacy rendering for
controlled comparisons; it does not change generator modes. In particular,
`original_only` describes canonical query generation, not a promise that the
backend submits every original word. arXiv discovery still loads the local NLP
model in that mode. Pipeline orchestration remains unchanged.

Dropping negation, comparisons, populations or conditions is a **discovery
relaxation**, never an equivalent scientific claim. Their exact omitted text
and the intact claim must be used during subsequent relevance assessment.

## Actual experiment

Used the existing ten claims, dates **2020-01-01–2026-10-30**, top **10** records
per request, and **one request per claim per arm per provider**. This is a
backend ablation: intact original/legacy arXiv rendering versus the same
selected discovery concepts on both APIs. It is not an evaluation of changes
to the generator's existing conservative alternative construction. Both
generator modes' outputs are saved separately in the manifest.

All **40** requests succeeded: 20 per provider, zero retries. Order was
counterbalanced by claim; arXiv retained its three-second pacing. OpenAlex
used one-second waits between completed calls. Errors would remain separate
from zero results, and a first 429/401/403 would stop that provider. No new
providers, PDFs, ranking or full-text verification were involved.

| Provider / arm | Requests | Unique papers | Zero-result claims | Summed latency | Abstracts | PDF URL metadata |
|---|---:|---:|---:|---:|---:|---:|
| OpenAlex original | 10 | 100 | 0/10 | 17.02 s | 89/100 | 90/100 |
| OpenAlex concepts | 10 | 100 | 0/10 | 13.15 s | 81/100 | 59/100 |
| arXiv original | 10 | 10 | 9/10 | 29.51 s | 10/10 | 10/10 |
| arXiv concepts | 10 | 40 | 6/10 | 28.79 s | 40/40 | 40/40 |

Within-provider overlaps were **9 OpenAlex** and **5 arXiv** papers. Natural
two-query unions were 191 and 45 unique papers respectively, at twice the
per-claim request cost and up to twice the retrieved candidate budget. Across
providers, DOI/arXiv ID/conservative-title deduplication produced **110 original**
versus **138 concept** identities, overlap **14**, and union **234**. Combined
arms cost 20 requests each; the union cost 40. These are retrieval coverage
measurements, not accuracy or relevance measurements.

Latency sums include in-client pacing, not just wire time. Separate attempt
timings are archived. None of the parsed dates violated the specified window;
arXiv dates describe submissions, unlike some journal publication dates.

Examples: arXiv battery, dark-matter and GNN originals each had zero results,
while their concept queries returned ten. Six claims remained empty in both
arXiv arms. Exercise/memory returned ten in each arm with five overlapping;
five original-only papers were displaced. OpenAlex's concept arm had the same
100-paper total but substantially different identities and fewer PDF URLs.
The displaced papers could matter for relevance; no labels establish whether
either arm was better.

Limitations include the small English spaCy model, source-order selection,
omitted outcomes/populations, exact-phrase restrictions, limited biomedical
arXiv coverage, top-10 truncation and live API variability. In particular,
`elevated atmospheric CO2 concentrations` remains a restrictive exact phrase,
and the two-concept agriculture query drops wheat yield. No extraction failures
occurred in these ten cases. No Precision@K or Recall@K is reported without
ground-truth judgments.

## Saved evidence and reproduction

- [Detailed report with all ten queries and omitted spans](../artifacts/arxiv-concepts-20261031/report.md).
- [Full results, per-provider and per-claim comparisons](../artifacts/arxiv-concepts-20261031/results.json).
- [Manual assessment CSV](../artifacts/arxiv-concepts-20261031/manual_assessment.csv):
  **250 returned-record rows**, including titles, abstracts, original claims,
  queries, provider/rank/identity, DOI, dates, URLs and blank judgment columns.
  Rows retain claim/arm provenance and may share a deduplicated identity.
- `calls.jsonl` and `responses/*.body` in that directory retain actual request
  parameters, safe headers, authentication-presence checks, timings and bodies.
  All 40 SHA-256 values were verified. Supplied credentials are redacted.

```powershell
.venv\Scripts\python.exe scripts/compare_arxiv_concepts.py --run-live --output artifacts/arxiv-concepts-new-run
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

Use a new directory to preserve prior evidence. Offline report generation is
available through `write_artifacts()` using saved `results.json`; it makes no
API calls. Historical provider-diagnostic scripts explicitly retain literal
rendering so reruns do not silently adopt this new policy.

Validation: **196 pytest tests passed** in 5.01 seconds; **Ruff passed**. New
regressions cover source-term preservation, omitted conditions, missing-model
fallback, quoted phrases, numeric scientific names, unchanged generator modes,
outgoing date/limit parameters, identity overlap and failure denominators.
The final renderer reproduced all ten archived query plans offline.
