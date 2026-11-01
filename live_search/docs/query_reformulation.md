# Audited deterministic scientific query generation

The generator is an experimental retrieval aid, not a scientific claim
verifier. No retrieval improvement has been established. Ranking, PDF
verification, and LLM generation are not implemented by this change.

## Audit findings and separation of responsibilities

The previous builder wrapped almost every extracted noun phrase in quotes,
including individual generic nouns. It usually copied the original sentence
with different punctuation, creating unnecessary exact-phrase constraints.
Actual pre-change outputs were captured before editing the implementation in
`query_audit_before.json`; no legacy generator is duplicated in the codebase.

The architecture now separates three responsibilities:

1. `src/paper_search/linguistic.py` extracts POS-tagged noun chunks, entities,
   and exact character spans. It does not construct queries or add quotes.
2. `src/paper_search/query_construction.py` selects meaningful concepts and
   decides whether a genuinely useful, meaning-preserving alternative exists.
3. `src/paper_search/query_rendering.py` applies only backend-supported syntax
   and reports unavoidable provider limitations.

`query_processing.py` coordinates these stages. The existing `QueryGenerator`
interface and API-client method signatures remain compatible. The first query
is the exact original input, including case, punctuation, and whitespace.
`normalized_claim` is a separate internal working string, and extraction
character offsets refer to that string, not the possibly whitespace-rich
original. Only internal processing and transport rendering normalize spaces.

## Concept extraction and selection

The installed spaCy English pipeline provides POS tags, dependency noun chunks,
and NER spans. Surface-span recovery preserves scientific atoms when the model
splits identifiers or units. Literal source patterns also recover vitamin
labels and signed numerical quantities with common scientific units. They do
not expand abbreviations, replace terminology, or invent synonyms.

The construction policy prefers meaningful multiword phrases and technical
identifiers over isolated generic nouns. A small explicit generic-noun list
filters concept candidates, never words from the query itself. Measurement
heads such as `expression` or `risk` are combined with their adjacent `of`
referents: `expression of TNF-α` and `risk of COVID-19` are useful concepts,
while isolated `expression` and `risk` are not treated as separate keywords.
Adjacent technical identifiers and generic complements such as `HER2+ cells`
can also be grouped without changing any source words.

At least two meaningful concepts are required, including at least one
multiword phrase or technical expression. Outcome/population roles are not
validated biomedical labels; the English parser's syntactic/NER labels remain
visible and can be wrong. In an earlier example it labeled an age span DATE.

## Construction policy and deliberate abstention

The builder retains the entire scientific assertion connecting selected
concepts. It removes only an anchored, recognized affirmative reporting frame,
for example `Our results show that`, `This study demonstrates that`, or
`In this study, we demonstrate that`. It does not turn the claim into a list
of disconnected nouns, reorder scientific terms, remove predicates, or erase
conditions to manufacture a shorter query.

The reduction must remove at least three lexical tokens and at least 20% of
the original token count. This is a transparent, hand-chosen utility rule,
not a threshold optimized against retrieval judgments. Article-only edits,
automatic quotation, and other near-duplicates do not qualify.

No alternate query is emitted for direct claims that already express the
scientific assertion concisely. Negated, qualified, uncertain, unrecognized,
or unsafe reporting frames are not stripped. `Preliminary findings suggest`
is deliberately retained. Negation, modals, comparisons, quantifiers,
abbreviations, populations, numbers, signs, units, and scientific conditions
inside the assertion are never deleted by construction.

This is bounded, conservative reformulation, not general semantic compression.
It can miss other useful reformulations. That abstention is preferable to
inventing an alternative that weakens or changes the scientific assertion.

## Configuration and diagnostics

Install the model explicitly; generation never downloads weights:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,nlp-model]"
```

Use `DeterministicQueryGenerator.from_settings(Settings.from_environment())`.

| Configuration | Behavior |
| --- | --- |
| `PAPER_SEARCH_QUERY_MODE=original_only` | Exact original input only; bypass NLP |
| `PAPER_SEARCH_QUERY_MODE=original_plus_concepts` | Original plus at most one useful alternative |
| `PAPER_SEARCH_MAX_QUERIES=1` | Original only regardless of mode |
| `PAPER_SEARCH_MAX_QUERIES=2` | Default maximum budget |
| `PAPER_SEARCH_QUERY_MODEL=en_core_web_sm` | Default installed English model |

The original `generate()` returns `list[str]` as before.
`generate_with_details()` additionally reports extraction status, model version,
POS tags, source spans, selected concepts, construction decision/reason,
removed reporting prefix, and qualitative recall caveats. Missing weights,
failed extraction, invalid source spans, insufficient concepts, unsupported
frames, and near-duplicate alternatives all retain just the original.

No token truncation, stemming, transliteration, unsupported synonyms, broad
stopword filtering, or automatic retrieval fan-out is used. Both ablation
modes leave the existing one-query search pipeline behavior unchanged.

## Provider-specific rendering and quoting

Focused queries contain no new exact-match quotation marks. A phrase explicitly
quoted by the user is not discarded by scientific construction.

- OpenAlex receives plain focused text. Explicit user phrase delimiters are
  rendered using supported double quotes. Uppercase operator words appearing
  as claim text are kept literal, rather than interpreted as exclusions.
  [Official search syntax](https://help.openalex.org/api/searching/).
- arXiv applies its documented `all:` field/AND syntax. Ordinary single words
  are unquoted. Quotes are used only for explicit multiword phrases and
  syntax-sensitive atoms such as hyphenated identifiers, negative values,
  or units containing punctuation. Literal claim negation never becomes an
  `ANDNOT` exclusion. [Official API manual](https://info.arxiv.org/help/api/user-manual.html).
- Semantic Scholar relevance search receives plain text and does not get
  exact-match quotes. The endpoint documents no phrase-query syntax and warns
  about hyphenated terms. Internal letter hyphens render as spaces with an
  explicit warning; canonical spelling, signs, ranges, and exponents are kept
  separately. [Official Graph API schema](https://api.semanticscholar.org/graph/v1/swagger.json).

For example, `physical activity may improve working memory` stays unquoted on
OpenAlex and Semantic Scholar, and becomes `all:physical AND all:activity AND
all:may AND all:improve AND all:working AND all:memory` on arXiv. Existing user
input `"working memory"` uses phrase quotes only on the supporting APIs.
`render_query()` preserves exact canonical `original`, returns backend `text`,
and exposes limitations through `warnings`. It cannot guarantee how a remote
index matches scientific symbols, phrase boundaries, or logical polarity.

## Actual before/after report and recall risks

[The audit report](query_audit.md) shows exact original, before-focused, and
after-focused queries for **14 diverse scientific claims**. Eight receive a
useful unquoted alternative; six receive only their original. The detailed
JSON is `artifacts/query-audit.json`. The outputs were generated with spaCy
3.8.18 and `en_core_web_sm` 3.8.0, not manually invented examples.

No recall loss or improvement has been measured. Important qualitative risks:

- Retained conditions such as -5 °C, 500 cycles, 100 kHz, acidic solutions,
  coastal waters, older adults, or under-five children can restrict matches.
  These conditions already exist in the original; construction does not add
  them. Dropping them would weaken the claim and is not attempted here.
- Removing reporting words may broaden candidate sets and change service-side
  ordering. Under a fixed top-k budget, relevant papers can therefore leave
  the returned list even when fewer lexical constraints are applied.
- A service may tokenize scientific symbols, units, negation, or acronyms
  differently. Semantic Scholar's hyphen workaround is explicitly lossy.
- No unsupported synonyms are added, so papers using different terminology
  can still be missed. General-English parsing is not scientific NER or
  logical entailment analysis.

A future comparison must hold claims, dates, providers, retrieval/ranking,
request budgets, result limits, provenance tracking, and deduplication policy
constant. Evaluate against judged relevant papers before claiming improvement.

## Reproduce locally

```powershell
.\.venv\Scripts\python.exe scripts/query_audit.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

The audit uses captured baseline outputs and makes no API or PDF requests.
Regression tests cover all 14 claims, exact original preservation, concept
selection, negation/uncertainty safety, numeric conditions, unsupported/near-
duplicate alternatives, and provider-specific rendering. Model tests run when
the installed weights are available and otherwise explicitly skip.
