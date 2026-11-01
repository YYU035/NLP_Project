# OpenAIRE discovery experiment

Completed UTC: 2026-11-01T05:51:51.640346+00:00. OpenAlex/arXiv are saved 31 October results, not fresh requests.

## Protocol and API

Same ten claims, exact saved original and space-separated spaCy concept queries; inclusive dates 2020-01-01–2026-10-30; top 10, one page, native relevance order. No model/query regeneration. Original/concept order alternates by claim. Anonymous public API only, at most 20 physical attempts, zero retries, serialized starts at least three seconds apart; stop on the first failed query. No PDF/landing-page downloads or relevance scoring.

[OpenAIRE v3 search](https://graph.openaire.eu/docs/apis/graph-api/research-products/) supports keyword `search`, `type=publication`, date filters and pageSize 1–100. [Terms](https://graph.openaire.eu/docs/apis/terms/) allow free access: anonymous up to 60/hour, authenticated up to 7,200/hour. This bounded run uses no credentials or paid services. Coordinate any other users of the same anonymous quota.

[Query syntax](https://graph.openaire.eu/docs/apis/graph-api/research-products/filtering/) supports quoted Boolean operands; this experiment uses plain keyword input without invented synonyms or phrase constraints. The adapter quotes an entire literal input if it contains an uppercase reserved operator, so claim text cannot become NOT exclusion.

## Observed results

OpenAIRE: **20 logical calls / 20 HTTP attempts**, **0 failures**, **114 returned record instances**, **112 unique claim–paper pairs across its two query arms.

| Provider | Arm | Calls | Records | Unique papers | Valid zeros | Failures | Descriptions/abstracts | PDF URLs | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openalex | original | 10 | 100 | 100 | 0/10 | 0 | 89 | 90 | 17.02 |
| openalex | alternative | 10 | 100 | 100 | 0/10 | 0 | 81 | 59 | 13.15 |
| arxiv | original | 10 | 10 | 10 | 9/10 | 0 | 10 | 10 | 29.51 |
| arxiv | alternative | 10 | 40 | 40 | 6/10 | 0 | 40 | 40 | 28.79 |
| openaire | original | 10 | 14 | 14 | 7/10 | 0 | 13 | 0 | 35.48 |
| openaire | alternative | 10 | 100 | 100 | 0/10 | 0 | 85 | 8 | 23.33 |

**11 OpenAIRE responses are confirmed top-10 capped** because their reported total exceeds ten; query_diagnostics in results.json preserves those totals. Short/empty responses are not treated as API failures. Latency sums include pacing.

Against the saved **234**-pair OpenAlex/arXiv union, OpenAIRE overlaps **18** pairs and contributes **94 new pairs**; combined coverage is **328 pairs**. New pairs include **79 descriptions** and **6 explicit PDF URL metadata records**. These new pairs have no assigned relevance labels.

## Per-claim/provider comparison

Counts are saved returned records (top 10), not all indexed hits. `pending`/`not_attempted` and failures must not be read as zeros.

| Claim | Provider | Original returned | Concepts returned | Unique union | New vs saved two-provider union | Status O / C |
|---|---|---:|---:|---:|---:|---|
| c01 | openalex | 10 | 10 | 18 | — | success / success |
| c01 | arxiv | 0 | 0 | 0 | — | zero_results / zero_results |
| c01 | openaire | 1 | 10 | 11 | 8 | success / success |
| c02 | openalex | 10 | 10 | 20 | — | success / success |
| c02 | arxiv | 0 | 10 | 10 | — | zero_results / success |
| c02 | openaire | 0 | 10 | 10 | 10 | zero_results / success |
| c03 | openalex | 10 | 10 | 20 | — | success / success |
| c03 | arxiv | 0 | 0 | 0 | — | zero_results / zero_results |
| c03 | openaire | 0 | 10 | 10 | 10 | zero_results / success |
| c04 | openalex | 10 | 10 | 16 | — | success / success |
| c04 | arxiv | 0 | 0 | 0 | — | zero_results / zero_results |
| c04 | openaire | 0 | 10 | 10 | 10 | zero_results / success |
| c05 | openalex | 10 | 10 | 20 | — | success / success |
| c05 | arxiv | 0 | 10 | 10 | — | zero_results / success |
| c05 | openaire | 3 | 10 | 13 | 9 | success / success |
| c06 | openalex | 10 | 10 | 20 | — | success / success |
| c06 | arxiv | 0 | 0 | 0 | — | zero_results / zero_results |
| c06 | openaire | 0 | 10 | 10 | 9 | zero_results / success |
| c07 | openalex | 10 | 10 | 19 | — | success / success |
| c07 | arxiv | 0 | 0 | 0 | — | zero_results / zero_results |
| c07 | openaire | 0 | 10 | 10 | 7 | zero_results / success |
| c08 | openalex | 10 | 10 | 19 | — | success / success |
| c08 | arxiv | 0 | 10 | 10 | — | zero_results / success |
| c08 | openaire | 0 | 10 | 10 | 8 | zero_results / success |
| c09 | openalex | 10 | 10 | 19 | — | success / success |
| c09 | arxiv | 0 | 0 | 0 | — | zero_results / zero_results |
| c09 | openaire | 0 | 10 | 10 | 9 | zero_results / success |
| c10 | openalex | 10 | 10 | 20 | — | success / success |
| c10 | arxiv | 10 | 10 | 15 | — | success / success |
| c10 | openaire | 10 | 10 | 18 | 14 | success / success |

## Decision and limitations

Of the novel pairs, **4 descriptions exceed 12,000 characters**. new_pairs_preview.csv truncates these with an explicit flag; the full CSV, results and raw responses retain the complete text. Some descriptions are much longer than normal abstracts, so description availability is not abstract quality. Language codes are retained but can be undetermined.

Retain as an **optional candidate source pending manual relevance assessment**, rather than replacing OpenAlex/arXiv: it supplies new identities, but their usefulness is not established. Review new_pairs_for_assessment.csv before adding routine request cost.

Descriptions are not guaranteed abstracts. The adapter strips markup and retains the first nonempty description; it does not translate or infer scientific evidence. Metadata can be multilingual, incomplete, duplicated, or a publication subtype other than a research article. Explicit PDF-shaped URLs alone count as PDF metadata; DOI/OA flags do not prove downloadable full text.

Deduplication reuses DOI/arXiv IDs and conservative exact-title/year/author matching, with known conflicting identifiers kept separate. A pair is new only if absent for that same claim from **both arms of both saved providers**. Novelty is relative to those saved top-10 lists, not to either provider's complete corpus. Different run dates, index updates and default ranking prevent a simultaneous controlled provider-quality comparison. Top-10 lists may be truncated; raw header.numFound is archived. No precision, accuracy or recall gain is inferred from additional papers.

## Evidence

- run.json freezes claims, saved queries, dates and baseline hash.
- calls.jsonl and responses/*.body retain requests, errors, timing and raw responses with SHA-256 hashes; no credentials are supplied.
- results.json / per_claim.csv contain counts and comparisons.
- openaire_papers.csv retains all normalized OpenAIRE occurrences.
- new_pairs_for_assessment.csv has one row per novel claim–paper pair, title/description, URLs and all retrieval provenance. manual_relevance and manual_notes are intentionally blank.
- new_pairs_preview.csv has the same identities and provenance, with a 12,000-character description preview for easier review.

## Review notes and validation

- OpenAIRE concepts returned ten candidates for all six claims with empty saved
  arXiv concept results (c01/c03/c04/c06/c07/c09). This expands observed candidate
  coverage, not proven relevant evidence. Original OpenAIRE queries returned
  only 14 records in total, with seven valid zeros; concepts returned 100 with
  no zeros. All ten concept lists were capped, as was original c10.
- Inspect the two near-identical SiC short-circuit review titles in the new CSV:
  differing punctuation/identifier completeness means the conservative matcher
  can retain near-duplicates. The 94 new pairs are algorithmic identities,
  not manually verified distinct works. Some astronomy results are repository
  deposits with unusually long descriptions; their scientific usefulness is
  unassessed. Do not equate a publication record with a peer-reviewed article.
- The six novel pairs with explicit PDF URL metadata are a small subset of the
  94 new pairs. Other landing links may resolve to full text, but none was
  fetched. This limits any immediate full-text-verification benefit.
- Validation: 226 pytest tests passed in 7.27 seconds; Ruff passed. Tests include
  normalization, missing fields, arXiv identifier preservation, date filters,
  authentication, malformed-envelope failures, shared retry/pacing, hourly
  attempt limits, opt-in pipeline behavior, claim-specific deduplication and
  explicitly flagged description previews. All 20 raw-response hashes and
  identical top-10/date parameters were verified offline. Both novel-pair CSVs
  contain 94 unique pair rows with blank relevance fields.
