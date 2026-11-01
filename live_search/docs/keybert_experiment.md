# Optional local KeyBERT experiment

`KeyBERTQueryGenerator` implements the existing `generate(claim, max_queries=...)`
interface, retaining the exact original first. It changes neither spaCy nor
provider behavior and is not enabled by default. This experiment submits one
extracted query per method, not an original-plus-query union.

## Method

[KeyBERT](https://maartengr.github.io/KeyBERT/api/keybert.html) scores source
keyphrase candidates using document/keyphrase embeddings and cosine/MMR
selection. The free Apache-2.0
[all-MiniLM-L6-v2 model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
runs locally on CPU, pinned to revision
`2a7cca6ad7605f3183bce07f677bda658059bbe9`. It does not embed/rank papers or call
a hosted inference API.

Candidates are literal one-to-four-atom source spans. Small boundary/reporting
rules exclude malformed candidates; scientific identifiers and quantity/unit
atoms such as `TNF-α`, `100 kHz` and `-5 °C` remain intact. MMR uses diversity
0.3, followed by selection of two or three nonoverlapping phrases in source
order. No stemming, synonyms or spaCy extraction are used. Omitted spans and
keyphrase scores are recorded; these are not paper relevance scores.

The pre-retrieval inspection caught reporting language, a split unit and an
empty punctuation candidate. These were fixed **before any retrieval**. Two
unrun drafts remain archived; only `queries.json` was used for the single live
run. [The qualitative review](../artifacts/keybert-experiment-20261101/query_review.md)
records all ten final queries and their trade-offs. No search outcomes were
used to tune the final queries.

Weights hashes and exact versions are frozen in `queries.json`: KeyBERT 0.9.0,
Sentence Transformers 5.7.0, CPU PyTorch 2.15.0, Transformers 5.22.1,
scikit-learn 1.9.1 and NumPy 2.5.4. This optional setup is heavier than spaCy.

## Actual results

Ten claims; dates **2020-01-01–2026-10-30**; **top 10**; fresh searches for both
methods; counterbalanced order. Each provider received twenty requests, ten
per method. **All 40 succeeded, with no retries.** Only free public retrieval
was used. OpenAIRE remained within its [hourly quota](https://graph.openaire.eu/docs/apis/terms/).
No arXiv requests, PDFs or relevance judgments were performed.

| Provider | Method | Records / unique pairs | Between-method overlap | Abstracts/descriptions | PDF URLs |
|---|---|---:|---:|---:|---:|
| OpenAlex | spaCy | 100 / 100 | 22 | 81 | 59 |
| OpenAlex | KeyBERT | 100 / 100 | 22 | 86 | 75 |
| OpenAIRE | spaCy | 100 / 100 | 13 | 85 | 8 |
| OpenAIRE | KeyBERT | 81 / 81 | 13 | 81 | 6 |

OpenAIRE KeyBERT returned **2** candidates for c01, **6** for c03 and **3** for
c08; archived totals confirm those small pools. Added conditions/comparators
are plausible contributors, but their individual effects were not isolated.
Of 37 lists returning ten, 36 had reported totals above ten; one reported
exactly ten. Top-10 truncation limits comparisons.

Against the historical **328**-pair union, fresh retrieval added **120 new
pairs**: **118 KeyBERT-only**, one shared with fresh spaCy, and one spaCy-only.
Thus KeyBERT contributes **119 historically new pairs**, not 119 proven
relevant papers. Even identical c10 queries produced one new historical
identity, illustrating index/order variability rather than a query advantage.

Only the saved provisional 250-row labels are reused by claim/identity. The
known-label subset is biased toward earlier retrieval: OpenAlex has 94 known
labels for fresh spaCy versus 38 for KeyBERT; OpenAIRE has 17 versus 5. Do not
compare these partial counts as full precision or accuracy. The separate
30-example human file is not mixed into this source. All new pairs remain
unlabelled; existing labels are provisional AI-assisted judgments.

**Retain spaCy as default and proceed with integration.** Keep KeyBERT optional
until new relevance assessment establishes a meaningful benefit. It supplies
novel candidates and better OpenAlex PDF metadata, but more identities do not
establish better relevance, and more specific queries can reduce coverage.

## Outputs and reproduction

- [Main report](../artifacts/keybert-experiment-20261101/report.md): per-claim
  counts, overlaps, request budgets and provisional-label coverage.
- `per_query.csv` / `results.json`: uniqueness, metadata, requests, failures,
  latency including pacing and known/unknown labels.
- `request_diagnostics.csv`: reported totals and capped-list flags.
- [New-pair CSV](../artifacts/keybert-experiment-20261101/new_pairs_for_assessment.csv)
  and [preview](../artifacts/keybert-experiment-20261101/new_pairs_preview.csv):
  claims, identities, titles/descriptions and all retrieval provenance. Prioritize
  `keybert_present=True`, `spacy_present=False`. New relevance fields are blank.
- `retrieved_papers.csv`: all 381 occurrences, including reused label provenance.
- `queries.json`, `calls.jsonl`, `responses/*.body`: frozen model/input details
  and response evidence. The two earlier query drafts were not live experiments.

Preview descriptions are truncated at 12,000 characters with an explicit flag;
the full CSV retains complete metadata. Save edited assessments under a new
filename before regenerating reports, which replace the blank-label exports.

```powershell
# Free CPU install, then optional project dependencies.
.venv\Scripts\python.exe -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e '.[dev,keybert]'

# Explicit public weights download; extraction thereafter is local-only.
.venv\Scripts\python.exe scripts/experiment_keybert.py download-model
.venv\Scripts\python.exe scripts/experiment_keybert.py prepare --output artifacts/keybert-new-run
# Inspect query_review.md before opting into a NEW experiment:
.venv\Scripts\python.exe scripts/experiment_keybert.py run --run-live --output artifacts/keybert-new-run

# Existing-run report regeneration only: no API/model requests.
.venv\Scripts\python.exe scripts/experiment_keybert.py report
```

The small general-English model and fixed candidate rules may miss scientific
entities or qualifiers. Deduplication can leave near-duplicates; OpenAIRE
descriptions may be multilingual, long or not abstracts. PDF URLs are metadata,
not verified downloads. No corpus recall, statistical significance or validated
relevance improvement is claimed.
