# arXiv concept-rendering comparison — 31 October 2026

Coverage only: there are no relevance labels or accuracy metrics.

## Protocol

Ten fixed claims; 2020-01-01–2026-10-30; top 10 per query. Each provider has a hard budget of 20 HTTP attempts, zero retries, one original and one concept request per claim. Order alternates by claim. Stop that provider after 429/401/403. No PDF downloads.

The original arm sends the intact claim to OpenAlex and the legacy every-token AND rendering to arXiv. The concept arm uses the same NLP-selected surface phrases on both providers: space-separated plain text on OpenAlex, supported phrase syntax on arXiv. This is a backend-rendering ablation, not a change to QueryGenerator or its original_only/original_plus_concepts modes. Both generator outputs are saved in results.json; they are not additional API requests.

The matched comparison is original versus concept, one request each. Their union is the natural extra-query case: two requests per claim, potentially 20 candidates rather than 10. It is not an equal-cost gain. Latency sums include pacing; attempt timings separate HTTP time. Shared originals are counted once in the union.

Identities use the existing DOI/arXiv ID/conservative exact-title matcher, not fuzzy matching. Counts exclude failed requests from valid-zero denominators. Metadata URLs are unverified.

## Per-provider coverage

| Provider | Arm | Requests | Unique | Zeros/valid | Failures | Seconds | Abstracts/records | PDF URLs/records |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| openalex | original | 10 | 100 | 0/10 | 0 | 17.02 | 89/100 | 90/100 |
| openalex | alternative | 10 | 100 | 0/10 | 0 | 13.15 | 81/100 | 59/100 |
| arxiv | original | 10 | 10 | 9/10 | 0 | 29.51 | 10/10 | 10/10 |
| arxiv | alternative | 10 | 40 | 6/10 | 0 | 28.79 | 40/40 | 40/40 |

openalex: overlap **9**, concept-only **91**, original-only **91**, two-query union **191** unique papers.


arxiv: overlap **5**, concept-only **35**, original-only **5**, two-query union **45** unique papers.

Combined across providers after deduplication: original **110**, concepts **138**, overlap **14**, union **234**. Each matched arm costs 20 total requests; the union costs 40.

## Per-claim matched comparison

| Provider | Claim | Original unique | Concepts unique | Overlap | Concept-only | Original-only | Union |
|---|---|---:|---:|---:|---:|---:|---:|
| openalex | c01 | 10 | 10 | 2 | 8 | 8 | 18 |
| openalex | c02 | 10 | 10 | 0 | 10 | 10 | 20 |
| openalex | c03 | 10 | 10 | 0 | 10 | 10 | 20 |
| openalex | c04 | 10 | 10 | 4 | 6 | 6 | 16 |
| openalex | c05 | 10 | 10 | 0 | 10 | 10 | 20 |
| openalex | c06 | 10 | 10 | 0 | 10 | 10 | 20 |
| openalex | c07 | 10 | 10 | 1 | 9 | 9 | 19 |
| openalex | c08 | 10 | 10 | 1 | 9 | 9 | 19 |
| openalex | c09 | 10 | 10 | 1 | 9 | 9 | 19 |
| openalex | c10 | 10 | 10 | 0 | 10 | 10 | 20 |
| arxiv | c01 | 0 | 0 | 0 | 0 | 0 | 0 |
| arxiv | c02 | 0 | 10 | 0 | 10 | 0 | 10 |
| arxiv | c03 | 0 | 0 | 0 | 0 | 0 | 0 |
| arxiv | c04 | 0 | 0 | 0 | 0 | 0 | 0 |
| arxiv | c05 | 0 | 10 | 0 | 10 | 0 | 10 |
| arxiv | c06 | 0 | 0 | 0 | 0 | 0 | 0 |
| arxiv | c07 | 0 | 0 | 0 | 0 | 0 | 0 |
| arxiv | c08 | 0 | 10 | 0 | 10 | 0 | 10 |
| arxiv | c09 | 0 | 0 | 0 | 0 | 0 | 0 |
| arxiv | c10 | 10 | 10 | 5 | 5 | 5 | 15 |

## Interpretation examples

Use the paired table to identify expanded coverage (for example an original zero becoming a nonzero concept result), unchanged coverage, and displacement (original-only identities). Expanded coverage is not relevance improvement. Original-only identities are candidates to inspect for possible recall loss. PDF metadata completeness can also deteriorate despite additional identities.

## Rendered arXiv queries and omitted text

- **c01:** `all:"TNF-α" AND all:"COVID-19"`
  Omitted: ["The results of this study show that the expression of ", " does not increase the risk of ", " in older adults."]
- **c02:** `all:"lithium-ion batteries" AND all:capacity`
  Omitted: ["In this study, we demonstrate that ", " retain 90% ", " after 500 cycles at -5 °C."]
- **c03:** `all:"SiC MOSFETs" AND all:"silicon devices"`
  Omitted: ["Our experiments show that the efficiency of ", " is not lower than that of ", " at 100 kHz."]
- **c04:** `all:"elevated atmospheric CO2 concentrations" AND all:"ocean acidification"`
  Omitted: ["This study demonstrates that ", " increase ", " in coastal waters."]
- **c05:** `all:"dark matter" AND all:"rotational velocity"`
  Omitted: ["The presence of ", " increases the ", " of spiral galaxies."]
- **c06:** `all:"deficit irrigation" AND all:"water use"`
  Omitted: ["Experimental results show that ", " reduces ", " without decreasing wheat yield under drought conditions."]
- **c07:** `all:"vitamin D supplementation" AND all:"respiratory infections"`
  Omitted: ["Preliminary findings suggest that ", " may reduce ", " in older adults."]
- **c08:** `all:"graph neural networks" AND all:"molecular property prediction"`
  Omitted: ["Our results show that ", " improve ", " compared with random forests."]
- **c09:** `all:"platinum nanoparticles" AND all:"hydrogen evolution"`
  Omitted: ["It has been demonstrated that ", " accelerate ", " at 25 °C in acidic solutions."]
- **c10:** `all:Exercise AND all:memory`
  Omitted: [" improves ", "."]

## Limitations

Selected phrases are discovery constraints, not equivalent claims. Negation, comparisons, populations and numerical conditions may be omitted and must be assessed against the intact claim. Keeping only two concepts can lose important outcomes; exact phrases can miss wording variants. spaCy's small English model is not scientific NER. There are no synonym expansions or model downloads at runtime.

More candidates or PDFs do not imply relevance. Original-only papers identify possible displacement under the matched budget, not proven relevance loss. Truncation, API variability, caching and different index/date semantics limit interpretation (arXiv uses submission dates). No relevance judgments, ranking or full-text verification were performed. See manual_assessment.csv for titles, abstracts and blank judgment columns, calls.jsonl for failures and attempts, and responses/ for credential-redacted raw responses.
