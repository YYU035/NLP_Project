# KeyBERT versus spaCy retrieval

## Method

Same ten original claims; dates 2020-01-01–2026-10-30; top 10; fresh spaCy and KeyBERT searches on OpenAlex/OpenAIRE, one request per method/claim/provider, counterbalanced order. Maximum 20 physical attempts per provider, no retries; a first failure stops that provider. No ranking, paid inference, PDFs or LLM judgments.

KeyBERT uses pinned local all-MiniLM-L6-v2 CPU embeddings, source-preserving 1–4-word candidates, cosine/MMR selection (diversity 0.3) and at most three nonoverlapping phrases. It retains the original claim through the existing QueryGenerator interface but this experiment submits only the extracted alternative. spaCy uses the exact saved concept queries; its behavior is unchanged. See query_review.md, generated before retrieval, for the ten-query qualitative comparison.

## Provider totals

| Provider | Method | Requests | Records / unique pairs | New vs history | Abstracts/descriptions | PDF URLs | Zeros | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| openalex | spacy | 10 | 100 / 100 | 2 | 81 | 59 | 0 | 0 |
| openalex | keybert | 10 | 100 / 100 | 60 | 86 | 75 | 0 | 0 |
| openaire | spacy | 10 | 100 / 100 | 0 | 85 | 8 | 0 | 0 |
| openaire | keybert | 10 | 81 / 81 | 66 | 81 | 6 | 0 | 0 |

openalex: overlap **22**, KeyBERT-only **78**, spaCy-only **78** pairs.

openaire: overlap **13**, KeyBERT-only **68**, spaCy-only **87** pairs.

Historical union: **328 claim–paper pairs** from both old query arms of OpenAlex/arXiv/OpenAIRE. Fresh retrieval contributes **120 new pairs**, of which **119 occur in KeyBERT** (including pairs shared with fresh spaCy).

| Provider | Claim | spaCy records / unique | KeyBERT records / unique | Overlap | KB-only | KB-only new vs history |
|---|---|---:|---:|---:|---:|---:|
| openalex | c01 | 10/10 | 10/10 | 3 | 7 | 4 |
| openalex | c02 | 10/10 | 10/10 | 0 | 10 | 7 |
| openalex | c03 | 10/10 | 10/10 | 0 | 10 | 8 |
| openalex | c04 | 10/10 | 10/10 | 1 | 9 | 7 |
| openalex | c05 | 10/10 | 10/10 | 1 | 9 | 7 |
| openalex | c06 | 10/10 | 10/10 | 0 | 10 | 9 |
| openalex | c07 | 10/10 | 10/10 | 1 | 9 | 8 |
| openalex | c08 | 10/10 | 10/10 | 2 | 8 | 5 |
| openalex | c09 | 10/10 | 10/10 | 4 | 6 | 4 |
| openalex | c10 | 10/10 | 10/10 | 10 | 0 | 0 |
| openaire | c01 | 10/10 | 2/2 | 0 | 2 | 1 |
| openaire | c02 | 10/10 | 10/10 | 0 | 10 | 10 |
| openaire | c03 | 10/10 | 6/6 | 0 | 6 | 6 |
| openaire | c04 | 10/10 | 10/10 | 2 | 8 | 8 |
| openaire | c05 | 10/10 | 10/10 | 1 | 9 | 8 |
| openaire | c06 | 10/10 | 10/10 | 0 | 10 | 10 |
| openaire | c07 | 10/10 | 10/10 | 0 | 10 | 10 |
| openaire | c08 | 10/10 | 3/3 | 0 | 3 | 3 |
| openaire | c09 | 10/10 | 10/10 | 0 | 10 | 10 |
| openaire | c10 | 10/10 | 10/10 | 10 | 0 | 0 |

37 queries returned exactly ten and are potentially capped; 36 have archived API totals greater than ten. A zero result is valid only when the API call succeeded; failures/unattempted calls remain separate. Identical c10 queries are a repeated-query control, not evidence of a strategy difference.

## Relevance and decision

Only exact claim/identity matches inherit the existing provisional AI-assisted labels. They are not validated ground truth. New/unmatched papers stay unlabelled; comparing label counts on this selectively labelled subset cannot establish full retrieval quality. No scores or labels are invented.

| Provider | Method | Known relevant / evidence | Known labels | Unlabelled |
|---|---|---:|---:|---:|
| openalex | spacy | 92/49 | 94 | 6 |
| openalex | keybert | 36/26 | 38 | 62 |
| openaire | spacy | 17/10 | 17 | 83 |
| openaire | keybert | 5/4 | 5 | 76 |

**Retain spaCy as the default and proceed with integration.** Keep KeyBERT optional for this ablation until new candidate relevance is assessed. Additional identities establish coverage novelty, not a meaningful relevance advantage. Review new_pairs_preview.csv/full CSV, prioritizing keybert_present=True and spacy_present=False.

## Limits and evidence

The model is general English, not scientific NER; a small fixed candidate filter can omit useful qualifiers. Query similarity scores are not paper relevance. Different query lengths may affect conjunction matching. Top-10 truncation, native search semantics, index variability, conservative deduplication and incomplete provisional labels limit this small ten-claim study. OpenAIRE descriptions may be long, multilingual or not abstracts. PDF URLs are unverified.

queries.json freezes model revision, weights hashes, package versions, selection details, source hashes and baseline. calls.jsonl/responses retain failures and raw responses. per_query.csv/results.json contain counts, overlaps, metadata, requests and latency (including pacing). retrieved_papers.csv includes reused labels and provenance; new pair CSVs have blank manual labels and all retrieval provenance. Preview descriptions are truncated to 12,000 characters with a flag; full text remains in the full CSV and results. Neither list is full-text scientific verification.

## Observed interpretation and validation

- OpenAlex: KeyBERT retains the 100-candidate total, with 86 abstracts and 75
  PDF metadata records versus spaCy's 81 and 59. OpenAIRE: KeyBERT has 81
  candidates versus 100, with 6 PDF metadata records versus 8. These are
  metadata/coverage trade-offs, not validated relevance advantages.
- The historical union is 328 pairs. Of 120 new pairs, 118 occur only in fresh
  KeyBERT, one occurs in both fresh methods, and one only in fresh spaCy.
  KeyBERT therefore contributes 119 historically new pairs. Provider-level
  novelty counts overlap across providers and must not simply be added.
- The OpenAlex irrigation query recovers one previously labelled evidence paper
  with wheat/drought concepts, versus zero known evidence for fresh spaCy.
  Nine KeyBERT results for that claim remain unlabelled, so this is an example
  to inspect, not evidence of an overall precision improvement.
- All 40 response hashes, exact planned query inputs and top-10/date parameters
  were verified offline. 242 pytest tests passed in 9.46 seconds; Ruff passed.
  New tests cover original retention, scientific atoms and complete units,
  malformed candidates, query limits, local-only loading, overlap, label reuse
  and changed-input rejection. Existing provider/spaCy behavior is unchanged.
