# Offline abstract-quality assessment

This optional sidecar flags suspicious stored abstracts for review. It does not
contact APIs, filter papers, change text, select replacement abstracts, or affect
ranking and verification. Existing retrieval and frontend schemas are unchanged.

`likely_valid` means no warning was detected by these rules. It does **not**
verify abstract completeness, relevance, factual accuracy or evidence sufficiency.

## Rules (abstract-quality-v1)

| Category | Rule |
|---|---|
| `missing` | Absent, empty or whitespace-only text. |
| `body_like` | More than 1,000 whitespace-delimited words, or more than 300 words with at least two distinct body-section signals. |
| `suspected_incomplete` | Continuation opening (and/or/but/whereas) plus “respectively” near the start; or trailing ellipsis after a dangling function word. |
| `uncertain` | Placeholder, explicit repeated abstract/translation headings, replacement character, other trailing ellipsis, or unclear sentence structure. |
| `likely_valid` | None of the preceding warnings. |

Rules are applied in the order above. Body signals include references, funding,
acknowledgements, conflicts of interest, ethics approval and author contributions.
Background/Methods/Results/Conclusions headings alone are not body signals.
Short length, missing final punctuation, scientific symbols and non-English
characters alone never establish incompleteness. A complete short summary can
receive `likely_valid` even though its publication type remains unknown.

No language model, language detector or new dependency is used. Explicit headings
can flag possible translations; fluent multilingual descriptions without headings
can be missed. Long genuine abstracts can be flagged. These are unvalidated
review heuristics, not a classifier with established accuracy.

## Usage

From `live_search/`, using the existing installed project environment:

```bash
python scripts/assess_abstract_quality.py --input artifacts/depth-approved-run --output outputs/abstract-quality-review --depths 10 30
```

The output directory must not exist and must be outside the input directory.
The script only reads saved `cNN.json` files. It assesses individual provider
occurrences and selected deduplicated abstracts separately. Top-10 prefixes use
the existing merge function on prefix records; top-30 uses saved final results.
Counts across depths or scopes must not be summed.

`assessment.csv` and `assessment.json` preserve exact stored text, source IDs,
canonical identities, aliases and query/rank provenance. JSON includes input
SHA-256 hashes. `report.md` compares nonempty availability with category counts
and lists nonempty flagged records. Source archives are never written to.
The text preserved is normalized provider text already stored by retrieval;
earlier parser normalization cannot be reversed by this assessment.

For programmatic use, `paper_search.abstract_quality.assess_abstract(text)`
returns a separate assessment; `assess_retrieval(result, claim_id="c01")`
returns separate provider-record and deduplicated assessments without mutating
`result`. Provider occurrences can include the same paper more than once.
Merged provenance retains all sources; its selected provider is not an exclusive
contribution attribution. Quality flags are never used by production retrieval.

Run focused offline tests with:

```bash
python -m pytest tests/test_abstract_quality.py
```

Tests cover saved OpenAlex continuation/long-description records, long OpenAIRE
descriptions, short scientific summaries, provenance and unchanged serialization.
