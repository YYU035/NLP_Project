# Provider diagnostics — 30 October 2026

Retain OpenAlex as the broad discovery source. Retain arXiv for its subject
strengths, but evaluate a less restrictive discovery policy before relying on
claim-length queries. Keep Semantic Scholar optional until an approved key can
be tested. The two recommended additions are **Europe PMC for life-science
discovery and eligible XML full text**, and **Crossref for DOI metadata enrichment**.
These are recommendations, not implemented provider changes.

No retrieval-accuracy improvement is established. Search hits, accessible text,
and support for a scientific claim are different outcomes.

## Evidence and reproduction

The starting evidence is the saved
[ten-claim experiment](../artifacts/query-evaluation-20261030/report.md) and its
[results](../artifacts/query-evaluation-20261030/results.json). This diagnosis
uses three representative saved claims: c01 (TNF-α/COVID-19/older adults), c02
(lithium-ion battery capacity under specified conditions), and c08 (graph neural
networks versus random forests for molecular property prediction).

The opt-in [diagnostic script](../scripts/provider_diagnostics.py) reuses existing
configuration, arXiv transport/pacing, query rendering, Atom parsing and the
experiment response recorder. Alternative endpoints are bounded diagnostic
requests, not new production clients. No main pipeline, ranking, PDF verification
or LLM generation code was changed.

Actual live runs made **42 HTTP requests**: arXiv 33, Semantic Scholar 1,
Europe PMC 3 (including one XML request), CORE 1, OpenAIRE 2, Crossref 2.
Each had one attempt. arXiv requests were serialized with the existing
three-second start interval. Alternative requests were sequential, generally
1.5 seconds apart within the alternative test sequence; Crossref used a
two-second interval. Timeout was 20 seconds. Each provider group stopped on
its first 429, 401 or 403. No repeated Semantic Scholar or CORE probes were made.

Saved evidence:

- [Main JSON](../artifacts/provider-diagnostics-20261030/summary.json),
  [CSV](../artifacts/provider-diagnostics-20261030/summary.csv), and
  [request log](../artifacts/provider-diagnostics-20261030/requests.jsonl).
- [arXiv follow-up JSON](../artifacts/provider-diagnostics-followup-20261030/summary.json)
  and [CSV](../artifacts/provider-diagnostics-followup-20261030/summary.csv).
- Both directories contain `responses/*.body`: original JSON/XML/error bodies,
  linked by request ID, file path and SHA-256 in each observation. Parsed records
  retain returned titles, abstracts/descriptions, dates and link fields.
- The recorder stores safe response headers and authentication-presence checks,
  not credential values or authorization headers. No Semantic Scholar, CORE or
  OpenAIRE key/token was configured. Raw responses contained no supplied secrets.
  `credential_redaction: false` means no supplied secret needed replacement,
  not that request authorization headers were exported.

Reproduce deliberately, using new output directories to preserve the originals:

```powershell
.venv\Scripts\python.exe scripts/provider_diagnostics.py --run-live --output artifacts/provider-diagnostics-rerun
.venv\Scripts\python.exe scripts/provider_diagnostics.py --run-live --arxiv-followup --output artifacts/provider-diagnostics-followup-rerun
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

The script fixes the historical date window to 2020-01-01 through 2026-10-30.
The second command only probes arXiv. Network access is required for the first
two commands; importing the script and running its unit tests performs no live
requests. Avoid rerunning the first command while a provider is throttling.

## A. arXiv

### Syntax audit

`render_query()` in [query_rendering.py](../src/paper_search/query_rendering.py)
joins claim tokens using `all:... AND all:...`. Explicit multiword phrases and
syntax-sensitive technical expressions receive quotes; ordinary words do not.
The original claim remains intact at the query-generator interface, but the
backend representation requires every rendered token, including reporting
language, conditions and numbers. Negation is searched as text, not evaluated
as a scientific relation. `all:` searches metadata fields, not article bodies.
The field prefixes, Boolean conjunction, phrase quoting and UTC
`submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]` syntax agree with the
[official API manual](https://info.arxiv.org/help/api/user-manual.html).

`ArxivClient.search()` in [arxiv.py](../src/paper_search/clients/arxiv.py) adds
`(rendered_query) AND submittedDate:[202001010000 TO 202610302359]` for this
experiment. This is a **submission-date** filter, not necessarily journal
publication date; returned `published` is the arXiv publication timestamp.
Pacing follows the [API terms](https://info.arxiv.org/help/api/tou.html).

For example, the current battery rendering includes
`all:In AND all:this AND all:study ... AND all:"lithium-ion" ... AND all:"90%"`
`... AND all:500 ... AND all:"-5" AND all:"°C"`.
The exact complete expressions and encoded request parameters are saved in
the logs. There were no HTTP failures or error Atom entries in the 33 probes.

### Positive controls and date isolation

| Control | Total hits | Returned | Interpretation |
|---|---:|---:|---|
| `id_list=1706.03762` | 1 | 1 | Existing parser retrieved *Attention Is All You Need*, dated 2017, with abstract/PDF URL |
| Same ID intersected with the 2020–2026 submission filter | 0 | 0 | Known excluded item confirms date filtering works |
| `ti:"Attention Is All You Need"`, all dates / filtered | 36 / 33 | 3 / 3 | Title phrase also matches derivative titles; it is not an exact-identity control |
| `all:electron`, filtered | 73,718 | 3 | Broad scientific search works |

### Controlled query comparison

Each cell gives server-reported total hits, **filtered / all years**. Positive
queries returned at most three records. Counts are not unique-paper counts or
relevance judgments. Both original and saved focused queries were reproduced.

| Saved claim | Current original | Saved focused | Broad keywords | Concept phrases |
|---|---:|---:|---:|---:|
| c01 TNF-α/COVID-19 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| c02 battery capacity | 0 / 0 | 0 / 0 | 236 / 292 | 207 / 258 |
| c08 molecular prediction | 0 / 0 | 0 / 0 | 363 / 388 | 154 / 159 |

Exact diagnostic alternatives before existing arXiv rendering:

| Claim | Broad text | Phrase text | Intentional relaxation |
|---|---|---|---|
| c01 | `TNF-α COVID-19` | `"expression of TNF-α" "COVID-19"` | Drops negated risk assertion and older-adult population |
| c02 | `lithium-ion batteries capacity` | `"lithium-ion batteries" capacity` | Drops retention assertion, 90%, 500 cycles and −5 °C |
| c08 | `graph neural networks molecular property prediction` | `"graph neural networks" "molecular property prediction"` | Drops improvement assertion and random-forest comparator |

For example, c08 phrase rendering becomes
`all:"graph neural networks" AND all:"molecular property prediction"`.
These alternatives preserve their remaining literal terms but are **not
meaning-equivalent rewrites**. They are diagnostic topic searches only.
They must not silently replace the original claim or its verification criteria.

Additional c01 sensitivity tests found 9,118 filtered hits for `COVID-19`,
15 all-years hits for literal `TNF-α`, and 68 for `TNF`. `TNF COVID-19`
returned one filtered record: *Prediction of repurposed drugs for treating lung
injury in COVID-19*. Its abstract contains “TNF signaling”; this does not establish
the original negated, population-specific claim. Dropping α relaxes target
specificity: TNF is not asserted to be an equivalent synonym for TNF-α.
No spelling expansion or transliteration was silently introduced.

**Diagnosis:** valid syntax, connectivity and parsing are demonstrated. Requiring
the complete claim is a strong explanation for battery/GNN zeros: topic and
phrase queries retrieve records in the same date window. Removing dates alone
does not resolve any of the three original/focused zeros. Dates reduce the
nonzero candidate pools, but are not the principal failure here. The c01
intersection stays empty even after substantial relaxation; narrower biomedical
coverage and index/notation matching remain plausible. These tests cannot
separate their contributions or prove absence of relevant papers from arXiv.
More hits after dropping conditions do not prove better relevance.

## B. Semantic Scholar

`Settings.from_environment()` reads `SEMANTIC_SCHOLAR_API_KEY`; it was **absent**.
`SemanticScholarClient.auth_headers()` correctly supplies `x-api-key` when a key
exists. The relevance `/graph/v1/paper/search` endpoint and requested fields
match the [official schema](https://api.semanticscholar.org/graph/v1/swagger.json).
That endpoint accepts plain search text, not the bulk-search Boolean syntax.
Existing rendering removes internal letter-hyphens as its documentation advises,
while keeping the canonical claim and warning about the transformation.

The [provider's API information](https://webflow.development.semanticscholar.org/product/api)
describes anonymous access as shared/throttled and recommends a key. Its shared
pool is not a guaranteed per-client allowance; introductory authenticated access
is described as 1 request/second. A key is not universally mandatory for this
endpoint, nor a guarantee of availability.

Inspection of [HTTPProvider](../src/paper_search/clients/http.py) found:

- One in-flight request per instance; Semantic Scholar starts are at least one
  second apart. Multiple instances/processes do not share that limiter.
- Default two retries after the initial request; exponential waits are 1 and
  2 seconds unless `Retry-After` requires longer.
- Both delta seconds and HTTP-date `Retry-After` are parsed. A requested delay
  over the configured 60-second ceiling causes failure rather than an early
  retry. No provider-wide circuit breaker suppresses later logical searches.

The previous experiment recorded **65 HTTP 429 responses**, with only three
valid outcomes among 24 logical searches after the separately documented
sparse-zero-response correction. This is a provider-availability problem;
429 must not count as zero retrieved papers. Repeated bounded retries across
new searches can still waste requests during sustained throttling.

This diagnosis made **one** anonymous, no-retry, plain `graph neural networks`
request (`limit=3`). It returned **429 in 0.52 seconds**, with a message advising
waiting or applying for a key. No `Retry-After` was recorded. Testing stopped;
no alternate route, IP, query flood or credential bypass was attempted.
There is no new successful Semantic Scholar metadata sample in this run.

**Operational recommendation:** obtain an approved key, then run a separate,
small authenticated test at its assigned allowance. Until then, reliable use
in this environment is unproven and the provider should remain optional. A
provider-wide cooldown/circuit breaker is a future design improvement, not a
confirmed isolated parsing bug fixed in this task.

## C. Alternative APIs: documented capabilities

Integration difficulty below is an engineering assessment, not a benchmark.
Availability of a URL does not establish access permission or successful download.

| API | Coverage and search | Authentication / limits | Abstracts and full text | Fit / complexity |
|---|---|---|---|---|
| Europe PMC | Life sciences/biomedicine; fielded Boolean/phrase queries, first-publication dates, OA filters | Anonymous REST worked; a current numeric REST quota was not verified in the consulted reference | `resultType=core` can supply abstracts and full-text URLs; eligible PMC articles expose XML | Strong biomedical complement; low–medium mapping effort, HTML abstracts and duplicate MED/PMC records need handling |
| CORE | Cross-disciplinary OA repository/journal aggregation; searchable works, v3 `q`; advanced grammar not independently validated here | Official page permits unregistered access under terms, recommends registration; advertises one batch or five single requests per 10 seconds, with account-specific tiers | Metadata and machine-readable full texts are advertised; availability/entitlements vary and were not live-validated | Potential full-text complement; medium effort plus quota/licensing validation; defer |
| OpenAIRE Graph | Multidisciplinary publications and linked research outputs; keyword search, type/date parameters, fielded Boolean filtering | Anonymous up to 60/hour; authenticated up to 7,200/hour, Bearer-token lifecycle | Optional descriptions and instance URLs/access rights; graph metadata is not a general article-body XML service | Viable broad metadata source; medium effort for nested versions, rights, language and identifiers |
| Crossref | Cross-disciplinary DOI deposits; bibliographic query, date/type filters, exact DOI lookup | Public keyless access; valid `mailto` enables polite pool. Current public list/search 1/s, single-record 5/s; polite 3/s and 10/s respectively | Optional deposited abstracts; publisher/TDM links and licenses, no hosted article-body service | Best DOI/date/license enrichment; low mapping effort; unsuitable alone for verification |

Official references supporting the table:

- Europe PMC [REST reference](https://dev.europepmc.org/RestfulWebService),
  [developer overview](https://dev.europepmc.org/developers) and
  [EMBL-EBI programmatic-access training](https://www.ebi.ac.uk/training/online/courses/embl-ebi-programmatically/europe-pmc-programmatically/).
  JSON/XML search and eligible `/PMCID/fullTextXML` are distinct operations;
  OA metadata does not guarantee that every returned record supplies XML.
- CORE [API service description](https://core.ac.uk/services/api) and
  [FAQ](https://core.ac.uk/faq). The detailed [v3 reference](https://api.core.ac.uk/docs/v3)
  was blocked to the documentation fetcher, so field-query grammar and current
  full-text entitlement details remain unconfirmed. The advertised quota is
  not a promise for the anonymous request observed here. Check applicable terms
  and registration eligibility before integration.
- OpenAIRE [Graph API overview](https://graph.openaire.eu/docs/apis/graph-api/overview/),
  [research products](https://graph.openaire.eu/docs/apis/graph-api/research-products/),
  [filtering](https://graph.openaire.eu/docs/apis/graph-api/research-products/filtering/),
  [authentication](https://graph.openaire.eu/docs/apis/authentication/) and
  [terms/limits](https://graph.openaire.eu/docs/apis/terms/). Tests used documented
  parameter-style **v3**, not deprecated v1/v2 or the separate v4 filter-style API.
  Descriptions should not automatically be labelled abstracts.
- Crossref [REST metadata](https://www.crossref.org/documentation/retrieve-metadata/rest-api/),
  [authentication](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/),
  [July 2026 request-type limits](https://community.crossref.org/t/refining-rest-api-limits-for-improved-stability-and-reliability/16137)
  and [text/data mining](https://www.crossref.org/documentation/retrieve-metadata/text-and-data-mining/).
  TDM links may require publisher authorization; a DOI link is not a PDF link.
  Current limits distinguish list requests from single DOI lookups.

### Observed live behavior

Dates were 2020-01-01 through 2026-10-30 where supported in these probes.
CORE's single access probe had no date constraint; Semantic Scholar's simple
connectivity probe also omitted dates. These are availability tests, not a
matched-budget provider-relevance comparison.

| Probe | Status | Total hits / returned | Text and date completeness in returned sample | Timing |
|---|---|---|---|---|
| Europe PMC `"TNF-α" AND "COVID-19"`, dates | 200 | 41,882 / 3 | Titles 3/3, abstracts 1/3, dates 3/3, PDF URLs 2/3 | 3.25 s |
| Europe PMC `exercise AND memory AND OPEN_ACCESS:Y`, dates | 200 | 69,085 / 3 | Titles 3/3, abstracts 2/3, dates 3/3, PDF URLs 3/3 | 0.74 s |
| Europe PMC `PMC13344646/fullTextXML` | 200 | One XML body | `article` root and body, 1,192,345 bytes; *Poster Session 1* | 4.61 s |
| CORE `q="graph neural networks"` | 429 | Failure, not zero | Empty error body; rate headers limit 10, remaining 0; interval unknown | 0.34 s |
| OpenAIRE v3 GNN, publication type/dates | 200 | 37,621 / 3 | Titles/descriptions/dates each 3/3; instance URLs saved | 1.58 s |
| OpenAIRE v3 TNF-α/COVID-19, publication type/dates | 200 | 2,346 / 3 | Titles/descriptions/dates each 3/3; multilingual record observed | 1.51 s |
| Crossref bibliographic battery capacity, dates | 200 | 418,150 / 3 | Titles/DOIs present, abstracts 0/3; all three were peer-review records | 0.96 s |
| Crossref exact saved DOI | 200 | 1 / 1 | Title/DOI/date/link/license metadata; no abstract | 0.64 s |

Timing is measured per diagnostic operation and includes parsing; arXiv operation
durations can include pacing waits. Individual HTTP response timings are also
recorded. Small samples and cached/proxied live infrastructure limit comparisons.

Examples worth inspecting in the saved responses:

- Europe PMC returned a TNF-α/IL-8 post-COVID venous-thromboembolism study, plus
  a near-duplicate MED/PMC title variant and a large poster-session record.
  This warns against treating every hit as a distinct, focused paper. The XML
  test selected the first eligible OA candidate, the poster session, not the
  exercise paper; it proves XML access for one record, not claim relevance.
- OpenAIRE returned *Methods for Constructing Graph Neural Networks* and a
  spatiotemporal-GNN survey. A Ukrainian biomedical title illustrates language
  handling needs. Instance access data were saved but links were not fetched;
  direct PDF usability and article-body availability are untested.
- Crossref's three battery-query results were review reports for the same
  manuscript topic, not three research articles. Exact DOI lookup recovered
  *Why does COVID-19 disproportionately affect older people?*. Querying without
  publication-type filtering is unsuitable for counting candidate papers.
- Across the first arXiv run, all 44 returned record instances had titles,
  abstracts, publication dates and PDF URLs;
  repeated date/query variants can return the same paper. This is not 44 unique
  papers. The saved abstracts and links permit later offline inspection.

CORE's 429 does not prove that authentication is mandatory, that its corpus is
empty, or that its service is generally unavailable. No current CORE result
schema, abstract completeness or full-text access was validated. Semantic
Scholar and CORE failures are explicitly stored with `result_count=null`.

## D. Decision and limits

| Provider | Recommendation | Evidence / next step |
|---|---|---|
| OpenAlex | Retain | Existing ten-claim experiment produced usable broad results; not retested here |
| arXiv | Retain with subject-aware expectations; evaluate query-policy repair | 33 valid responses; broader battery/GNN searches work, complete original/focused queries do not. Keep original as control and record every dropped condition |
| Semantic Scholar | Optional; defer reliance until authenticated validation | No key, prior sustained throttling, one new 429. Do not keep retrying anonymously |
| Europe PMC | **New integration 1: prioritize** | Biomedical coverage and successful metadata plus one eligible XML response; add explicit rights/source/duplicate handling |
| Crossref | **New integration 2: enrichment only** | Exact DOI lookup works; discovery sample included peer-review records and no abstracts |
| OpenAIRE | Defer as a possible later broad-source replacement | Working metadata API, but overlap with OpenAlex, language handling and actual full-text utility need evaluation |
| CORE | Defer | Full-text promise is relevant, but local access, quota and entitlements remain unvalidated |

No replacement is justified solely by this small test. Do not integrate more
than the two recommended additions before evaluating their distinct contribution.
No confirmed new isolated production bug required a fix in this diagnosis.

Limits: samples are truncated to three records; totals are provider-specific
and not deduplicated across APIs. Broad and phrase variants deliberately drop
constraints and may lower precision or miss technical variants. Search indexes
may tokenize symbols differently. arXiv submission dates differ from journal
dates, and other APIs expose differing date semantics. Metadata can be missing,
duplicated, multilingual, or refer to non-article outputs. XML/text access and OA
flags do not establish scientific support or redistribution rights. No relevance
labels, Precision@K, Recall@K, PDF downloads or claim verification were performed.
Live results, quotas and service availability may change; authenticated S2/CORE
behavior remains unknown. Official documentation inaccessible to the fetcher is
identified above rather than inferred from a third-party tutorial.

Validation: `.venv\Scripts\python.exe -m pytest` completed with **186 passed**
in 9.45 seconds (Python 3.13.5). This includes four new offline diagnostic tests
covering the finite request matrix, paired date constraints, saved-query reuse,
and the distinction between errors and valid empty responses.
`.venv\Scripts\python.exe -m ruff check .` completed with **All checks passed**.
An offline integrity check verified all **42** archived response SHA-256 values.
The combined observations comprise 23 successes (including one XML response),
17 valid zero-result responses and two HTTP failures; failures are not zeros.
