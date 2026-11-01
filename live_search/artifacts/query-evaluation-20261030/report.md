# Live scientific search: preliminary query comparison

Started (UTC): 2026-10-30T09:17:38.219524+00:00. Finished (UTC): 2026-10-30T09:19:29.020008+00:00.

Claims: 10; limit: 5 per query; dates: 2020-01-01 to 2026-10-30.

**Retrieval coverage only. No ground-truth relevance labels are available; Precision@K and Recall@K are not computed. Extra papers must not be interpreted as relevant papers or improved accuracy.**

## Live validation and provider coverage

The original query for each of the ten claims is the validation call. Successful empty responses are distinct from failures. Metadata counts below use original-call records, before deduplication.

| Provider | Original OK / failed | All calls OK / failed | HTTP requests | 429s | Original / reform observed unique | Added observed |
|---|---:|---:|---:|---:|---:|---:|
| openalex | 10 / 0 | 24 / 0 | 24 | 0 | 50 / 63 | 13 |
| arxiv | 10 / 0 | 24 / 0 | 24 | 0 | 5 / 5 | 0 |
| semantic_scholar | 1 / 9 | 3 / 21 | 68 | 65 | 0 / 0 | 0 |

Observed unique totals are deduplicated across claims per provider; missing provider calls make totals partial. Only complete per-claim pairs support additional/overlap comparisons below.

| Provider | Records | Titles | Authors | DOIs | Dates | Abstracts | PDF URLs | Date violations | Mean original latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openalex | 50 | 50 | 50 | 49 | 50 | 46 | 45 | 0 | 1.18 |
| arxiv | 5 | 5 | 5 | 1 | 5 | 5 | 5 | 0 | 3.09 |
| semantic_scholar | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4.56 |

Dates are provider publication dates (arXiv original submission dates). Missing dates remain unknown; PDF availability means URL metadata only. No PDFs or landing pages were downloaded.

## Per-provider, per-claim natural comparisons

| Provider | Claim | Status | Original unique | Reform union | Overlap | Added | Wire requests original / reform | Zero O / R |
|---|---|---|---:|---:|---:|---:|---:|---|
| openalex | c01 | complete | 5 | 6 | 5 | 1 | 1 / 2 | False / False |
| openalex | c02 | complete | 5 | 8 | 5 | 3 | 1 / 2 | False / False |
| openalex | c03 | complete | 5 | 8 | 5 | 3 | 1 / 2 | False / False |
| openalex | c04 | complete | 5 | 7 | 5 | 2 | 1 / 2 | False / False |
| openalex | c05 | complete | 5 | 5 | 5 | 0 | 1 / 1 | False / False |
| openalex | c06 | complete | 5 | 6 | 5 | 1 | 1 / 2 | False / False |
| openalex | c07 | complete | 5 | 5 | 5 | 0 | 1 / 1 | False / False |
| openalex | c08 | complete | 5 | 7 | 5 | 2 | 1 / 2 | False / False |
| openalex | c09 | complete | 5 | 6 | 5 | 1 | 1 / 2 | False / False |
| openalex | c10 | complete | 5 | 5 | 5 | 0 | 1 / 1 | False / False |
| arxiv | c01 | complete | 0 | 0 | 0 | 0 | 1 / 2 | True / True |
| arxiv | c02 | complete | 0 | 0 | 0 | 0 | 1 / 2 | True / True |
| arxiv | c03 | complete | 0 | 0 | 0 | 0 | 1 / 2 | True / True |
| arxiv | c04 | complete | 0 | 0 | 0 | 0 | 1 / 2 | True / True |
| arxiv | c05 | complete | 0 | 0 | 0 | 0 | 1 / 1 | True / True |
| arxiv | c06 | complete | 0 | 0 | 0 | 0 | 1 / 2 | True / True |
| arxiv | c07 | complete | 0 | 0 | 0 | 0 | 1 / 1 | True / True |
| arxiv | c08 | complete | 0 | 0 | 0 | 0 | 1 / 2 | True / True |
| arxiv | c09 | complete | 0 | 0 | 0 | 0 | 1 / 2 | True / True |
| arxiv | c10 | complete | 5 | 5 | 5 | 0 | 1 / 1 | False / False |
| semantic_scholar | c01 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 3 / 6 | unavailable / unavailable |
| semantic_scholar | c02 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 3 / 6 | unavailable / unavailable |
| semantic_scholar | c03 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 2 / 5 | True / unavailable |
| semantic_scholar | c04 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 3 / 6 | unavailable / unavailable |
| semantic_scholar | c05 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 3 / 3 | unavailable / unavailable |
| semantic_scholar | c06 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 3 / 6 | unavailable / unavailable |
| semantic_scholar | c07 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 3 / 3 | unavailable / unavailable |
| semantic_scholar | c08 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 3 / 6 | unavailable / unavailable |
| semantic_scholar | c09 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 3 / 6 | unavailable / unavailable |
| semantic_scholar | c10 | incomplete_provider_failure | unavailable | unavailable | unavailable | unavailable | 3 / 3 | unavailable / unavailable |

## Zero-result rates and measured costs

Rates below use successful claim outcomes only; unavailable calls are excluded, never counted as empty responses.

| Provider | Original zero rate | Reform union zero rate | Original latency sum (s) | Extra-query latency sum (s) |
|---|---:|---:|---:|---:|
| openalex | 0/10 (0%) | 0/10 (0%) | 11.79 | 7.44 |
| arxiv | 9/10 (90%) | 9/10 (90%) | 30.88 | 22.30 |
| semantic_scholar | 1/1 (100%) | unavailable | 45.58 | 34.85 |

## Per-claim coverage across available providers

All observations below are partial if a provider failed. Counts do not replace missing outcomes from unavailable providers.

| Claim | Status | Observed original unique | Observed reform unique |
|---|---|---:|---:|
| c01 | incomplete_provider_failure | 5 | 6 |
| c02 | incomplete_provider_failure | 5 | 8 |
| c03 | incomplete_provider_failure | 5 | 8 |
| c04 | incomplete_provider_failure | 5 | 7 |
| c05 | incomplete_provider_failure | 5 | 5 |
| c06 | incomplete_provider_failure | 5 | 6 |
| c07 | incomplete_provider_failure | 5 | 5 |
| c08 | incomplete_provider_failure | 5 | 7 |
| c09 | incomplete_provider_failure | 5 | 6 |
| c10 | incomplete_provider_failure | 10 | 10 |

## Matched-request-budget comparison and natural cost

The common original response is fetched once and reused by both modes. For claims with an alternative, matched budget uses two logical calls per arm: original + fresh repeated original versus original + concept query. Extra-call order is counterbalanced by claim index. With no alternative both arms use one call. Actual wire attempts can differ due to retries; those mismatches are reported, not called matched.

| Provider | Complete matched pairs | Logical / wire matched pairs | Original / reform unique sum | Lost from matched baseline sum | Natural extra wire requests | Repeat-control overhead |
|---|---:|---:|---:|---:|---:|---:|
| openalex | 10 | 10 / 10 | 50 / 63 | 0 | 7 | 7 |
| arxiv | 10 | 10 / 10 | 5 / 5 | 0 | 7 | 7 |
| semantic_scholar | 0 | 10 / 8 | unavailable / unavailable | unavailable | 21 | 18 |

Sums above are per-claim unique counts, not global unique papers. Natural union includes the identical original response, so losing original papers in that union is impossible by design. Fresh repeats can expose variability and may add papers to the matched baseline. The experiment's physical HTTP total counts each shared original only once. Per-arm latency is a sum of client-call durations, including pacing/retries, not independent wall-clock A/B experiments.

## Examples

- Helped observed coverage, not relevance: openalex c01; openalex c02; openalex c03.
- Made no observed coverage difference: openalex c05; openalex c07; openalex c10.
- Appeared harmful to matched coverage: not observed in complete pairs.
- Appeared harmful in extra cost only: arxiv c01; arxiv c02; arxiv c03.

## Actual API failures and parsing observations

- openalex: auth=anonymous; HTTP statuses {"200": 24}; errors {}.
- arxiv: auth=not_required; HTTP statuses {"200": 24}; errors {}.
- semantic_scholar: auth=anonymous; HTTP statuses {"429": 65, "200": 3}; errors {"Semantic Scholar returned HTTP 429; retries exhausted": 21}.
- 2 saved HTTP-200 Semantic Scholar responses reported total=0 and offset=0 without a data array. A confirmed client fix was verified against those exact archives offline. They are corrected to zero_results; original runtime statuses and errors remain in calls.jsonl and parser_correction. No new live requests were made for this correction.
- A successful Semantic Scholar repeat control parsed 5 real papers. It does not replace failed original/alternative outcomes or establish a complete paired Semantic Scholar comparison.

## Claims and generated queries

- **c01 (biomedical)** Original: The results of this study show that the expression of TNF-α does not increase the risk of COVID-19 in older adults.
  Alternative: the expression of TNF-α does not increase the risk of COVID-19 in older adults.
- **c02 (battery_engineering)** Original: In this study, we demonstrate that lithium-ion batteries retain 90% capacity after 500 cycles at -5 °C.
  Alternative: lithium-ion batteries retain 90% capacity after 500 cycles at -5 °C.
- **c03 (power_electronics)** Original: Our experiments show that the efficiency of SiC MOSFETs is not lower than that of silicon devices at 100 kHz.
  Alternative: the efficiency of SiC MOSFETs is not lower than that of silicon devices at 100 kHz.
- **c04 (climate)** Original: This study demonstrates that elevated atmospheric CO2 concentrations increase ocean acidification in coastal waters.
  Alternative: elevated atmospheric CO2 concentrations increase ocean acidification in coastal waters.
- **c05 (astronomy)** Original: The presence of dark matter increases the rotational velocity of spiral galaxies.
  Alternative: none
- **c06 (agriculture)** Original: Experimental results show that deficit irrigation reduces water use without decreasing wheat yield under drought conditions.
  Alternative: deficit irrigation reduces water use without decreasing wheat yield under drought conditions.
- **c07 (nutrition_uncertainty)** Original: Preliminary findings suggest that vitamin D supplementation may reduce respiratory infections in older adults.
  Alternative: none
- **c08 (computer_science)** Original: Our results show that graph neural networks improve molecular property prediction compared with random forests.
  Alternative: graph neural networks improve molecular property prediction compared with random forests.
- **c09 (catalysis)** Original: It has been demonstrated that platinum nanoparticles accelerate hydrogen evolution at 25 °C in acidic solutions.
  Alternative: platinum nanoparticles accelerate hydrogen evolution at 25 °C in acidic solutions.
- **c10 (general_science)** Original: Exercise improves memory.
  Alternative: none

## Limitations and reproducibility

- One truncated page per query; at most the configured result limit. Provider ordering, index updates, proxy caches, and quota state vary.
- No labels, relevance judgments, significance tests, or accuracy conclusions. Gains mean extra identities only; zero results do not refute a claim. Removing framing changes service-side top-k ordering.
- DOI and arXiv identifiers are used when exposed by metadata. Exact long-title matching additionally requires the same known year and a shared full author; conflicting identifiers do not merge. Missing identifiers may leave duplicates; no fuzzy title matching is used.
- Provider failures and rate limits are unavailable outcomes, not zero results. Zero-rate denominators exclude failed calls; detailed counts/denominators are in provider_summary.csv and results.json.
- Authentication mode/header checks are recorded without secrets. A keyless run cannot validate keyed account access or quota increases.
- Raw HTTP bodies, safe headers, exact parameters, timings, hashes and normalized papers are saved locally. Response bodies are decoded by httpx; credentials reflected in bodies are redacted and flagged.
- results.json contains per-claim cross-provider observed coverage, identity groups, and all IDs. calls.csv/papers.csv/comparisons.csv provide detailed exports. calls.jsonl is checkpointed after each call.
- No LLM, BM25/SBERT ranking, PDF download, or verification is run.
