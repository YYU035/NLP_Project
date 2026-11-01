# arXiv spaCy versus KeyBERT: frozen-phrase comparison

## Method and translation

Same ten claims, dates 2020-01-01–2026-10-30, top 10, one request per method/claim. Order alternates by claim. At most 20 public arXiv requests, three-second minimum pacing, no retries; stop on a failed query. No OpenAlex/OpenAIRE, model inference, relevance API or PDF requests. Union is calculated offline.

**Important:** default ArxivClient concepts mode re-extracts spaCy concepts and could overwrite KeyBERT choices. This experiment explicitly uses the ALREADY EXISTING `query_strategy=literal` mode for BOTH methods. Frozen source phrases are individually quoted before the existing renderer joins them using AND. No production code/default changed. The spaCy wire expressions were checked against all ten saved 31 October expressions; they match exactly. KeyBERT uses the frozen previously tested phrases, not a spaCy re-selection or regenerated model output.

[Official arXiv syntax](https://info.arxiv.org/help/api/user-manual.html) supports all: fields, exact phrases and AND conjunctions. all: searches indexed metadata, not article-body full text. The full UTC submission-date suffix is shown below; it is not necessarily a journal publication-date restriction.

## Aggregate results

| Method | Records / unique pairs | Abstracts | PDF URLs | Known relevant / evidence | Unresolved |
|---|---:|---:|---:|---:|---:|
| spacy | 40 / 40 | 40 | 40 | 29 / 21 | 5 |
| keybert | 12 / 12 | 12 | 12 | 9 / 7 | 3 |
| union | 52 / 42 | 42 | 42 | 29 / 21 | 7 |

Overlap: **10**; KeyBERT-only: **2**; spaCy-only: **30**. Historically new pairs: **2**, relative to the full previous **448**-pair collection.

Union records count all returned occurrences; unique pairs are deduplicated by claim and the existing DOI/arXiv-ID/conservative-title matcher. Union needs no extra requests.

| Method | HTTP attempts | Valid zero queries | Failures | Latency sum |
|---|---:|---:|---:|---:|
| spacy | 10 | 6 | 0 | 27.43 s |
| keybert | 10 | 8 | 0 | 29.66 s |

## Per-claim results

T/U = returned/unique; R/E = known relevant/evidence; ? = unresolved. A failed/unattempted query is unavailable, not zero. See per_query.csv for API total matches and status.

| Claim | spaCy T/U; R/E; ? | KeyBERT T/U; R/E; ? | Union unique | Overlap |
|---|---:|---:|---:|---:|
| c01 | 0/0; 0/0; 0 | 0/0; 0/0; 0 | 0 | 0 |
| c02 | 10/10; 3/0; 1 | 0/0; 0/0; 0 | 10 | 0 |
| c03 | 0/0; 0/0; 0 | 0/0; 0/0; 0 | 0 | 0 |
| c04 | 0/0; 0/0; 0 | 0/0; 0/0; 0 | 0 | 0 |
| c05 | 10/10; 8/7; 2 | 2/2; 0/0; 2 | 12 | 0 |
| c06 | 0/0; 0/0; 0 | 0/0; 0/0; 0 | 0 | 0 |
| c07 | 0/0; 0/0; 0 | 0/0; 0/0; 0 | 0 | 0 |
| c08 | 10/10; 9/7; 1 | 0/0; 0/0; 0 | 10 | 0 |
| c09 | 0/0; 0/0; 0 | 0/0; 0/0; 0 | 0 | 0 |
| c10 | 10/10; 9/7; 1 | 10/10; 9/7; 1 | 10 | 10 |

## Exact submitted arXiv expressions

Every expression includes the actual date suffix. Source phrase lists and client-input text are frozen in run.json.

Each expression below was sent as `(<expression>) AND submittedDate:[202001010000 TO 202610302359]`. Complete wire strings are recorded in per_query.csv and calls.jsonl.

| Claim | spaCy expression | KeyBERT expression |
|---|---|---|
| c01 | `all:"TNF-α" AND all:"COVID-19"` | `all:"expression of TNF-α" AND all:"risk of COVID-19" AND all:"older adults"` |
| c02 | `all:"lithium-ion batteries" AND all:capacity` | `all:"lithium-ion batteries" AND all:"90% capacity" AND all:"-5 °C"` |
| c03 | `all:"SiC MOSFETs" AND all:"silicon devices"` | `all:"efficiency of SiC MOSFETs" AND all:"silicon devices" AND all:"100 kHz"` |
| c04 | `all:"elevated atmospheric CO2 concentrations" AND all:"ocean acidification"` | `all:"atmospheric CO2 concentrations" AND all:"ocean acidification" AND all:"coastal waters"` |
| c05 | `all:"dark matter" AND all:"rotational velocity"` | `all:"dark matter" AND all:rotational AND all:"velocity of spiral galaxies"` |
| c06 | `all:"deficit irrigation" AND all:"water use"` | `all:"deficit irrigation" AND all:"wheat yield" AND all:"drought conditions"` |
| c07 | `all:"vitamin D supplementation" AND all:"respiratory infections"` | `all:"vitamin D supplementation" AND all:"respiratory infections" AND all:"older adults"` |
| c08 | `all:"graph neural networks" AND all:"molecular property prediction"` | `all:"graph neural networks" AND all:"molecular property prediction" AND all:"random forests"` |
| c09 | `all:"platinum nanoparticles" AND all:"hydrogen evolution"` | `all:"platinum nanoparticles" AND all:"hydrogen evolution" AND all:"acidic solutions"` |
| c10 | `all:Exercise AND all:memory` | `all:Exercise AND all:memory` |
## Syntax diagnosis and decision

Quotes require phrase order; AND requires every quoted concept to match. KeyBERT commonly adds conditions or comparators (older adults, 90% capacity, -5 °C, 100 kHz, coastal waters, random forests, acidic solutions). These can plausibly exclude otherwise topical papers. A valid zero is not evidence that the scientific claim is false or that no relevant article exists outside indexed metadata.

No phrases, AND operators or dates were relaxed during the comparison, and no extra backoff search was made. Thus per-term causality, notation variants, date effects and corpus coverage cannot be separated. Compare paired outcomes as observations, not proof that quoting alone caused a zero. A ten-result list can be capped; inspect reported_total in per_query.csv.

**KeyBERT supplies additional arXiv identities, but their relevance must be assessed before claiming a benefit. Keep spaCy and use the deduplicated union only if its additional request/candidate cost is acceptable.**

A uniform two-generator interface is feasible with provider-specific syntax, but the current default arXiv concept renderer must not be assumed to preserve KeyBERT phrases. The explicit literal-mode binding tested here would need deliberate integration; production is unchanged. OpenAlex/OpenAIRE strategy findings do not prove equal benefit on arXiv.

## Labels, exports and limitations

Existing bulk250/new120/30-example assessments were frozen and matched by canonical claim–paper identity. Twelve cross-source conflicts remain unresolved, consistent with the previous final comparison. Missing labels are not irrelevant; all labels are provisional AI-assisted judgments. No labels are invented for new papers.

retrieved_papers.csv contains all occurrences, syntax and reused labels. new_pairs_for_assessment.csv contains only historically new pairs; unresolved_pairs_for_assessment.csv also covers earlier unlabelled/conflicting pairs. Both have blank manual-label columns and all retrieval provenance. Raw responses, errors and SHA-256 values are in calls.jsonl and responses/. Metadata links were not downloaded.

This small capped sample, native index variability, strict phrases and incomplete labels limit conclusions. There is no new relevance accuracy, significance, corpus recall, or full-text claim verification result.

## Observed syntax diagnosis and integration recommendation

- **c08 is a direct paired restriction example:** spaCy reported 154 matches;
  KeyBERT reported zero. Its wire expression is exactly the two spaCy phrases
  plus `AND all:"random forests"`, with the same dates. The added exact-phrase
  conjunct excludes the base candidates in this snapshot. This does not show
  that random forests are absent from relevant full texts, or isolate quoting
  from the choice of phrase itself.
- **c02:** spaCy reported 207 matches; KeyBERT zero. KeyBERT replaces the loose
  capacity term with `"90% capacity"` and adds `"-5 ?C"`. Their joint restrictive
  effect is observable; percentage versus temperature effects were not tested
  independently. Do not remove either silently or claim which one caused zero.
- **c05:** spaCy reported 104 matches and returned ten; KeyBERT reported and
  returned two different papers. Its rotational/velocity-of-spiral-galaxies
  wording is not a simple extra-conjunct subset of spaCy's rotational-velocity
  phrase. These two candidates are historically new and unlabelled, not
  demonstrated relevant gains. Both have abstracts and PDF URL metadata.
- spaCy zeros are c01/c03/c04/c06/c07/c09; KeyBERT adds c02/c08 to that zero set.
  The six shared zeros remain unresolved combinations of query restrictiveness,
  terminology, date window and provider coverage. None was an API failure.
  No zero-result spaCy claim was rescued by KeyBERT. No undated, OR, unquoted
  or relaxed diagnostic searches were made.
- Five ten-record lists have archived totals greater than ten; all are capped.
  KeyBERT c05's two-record list is not capped. c10's two wire queries and their
  ten returned papers are identical; this is an equal-budget duplicate control.
- The 42-pair union has 29 known relevant/21 evidence, the same known-labelled
  counts as spaCy alone; two added candidates have no labels yet. Seven union
  pairs are unresolved overall. Do not equate those unknowns with irrelevance.

**Final scope recommendation:** keep the previously supported spaCy+KeyBERT
combination on OpenAlex/OpenAIRE, but use **spaCy as the routine arXiv query**
and keep KeyBERT optional there pending assessment of its two new pairs. A
blanket second arXiv query costs ten additional calls here for two unassessed
pairs and rescues no zero-result claim, so uniform benefit across all three
providers is not established. Both can share the strategy interface while
using provider-aware execution policies. If a uniform two-query policy is
chosen for simplicity, deduplicate identical c10 requests and accept that its
arXiv relevance benefit remains unproven.

Any future arXiv KeyBERT integration must preserve its explicit phrase boundaries
using the tested existing literal-mode binding; passing its concatenated text
through default concepts mode would run spaCy again. This experiment deliberately
avoided that confound without modifying production defaults. Person 2 can rank
the deduplicated 42-pair candidate union from this run, with original claims and
strategy/provider/query provenance; provisional labels remain evaluation data.

Validation: **249 pytest tests passed** in 17.19 seconds; **Ruff passed**. All
20 HTTP responses were 200, with one attempt each. Their SHA-256 values, exact
frozen wire expressions, top-10 parameters and date suffixes were verified.
There were no paid services, model downloads, PDF downloads, production edits
or extra union/diagnostic search requests.
