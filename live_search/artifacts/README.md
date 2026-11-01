# Retrieval experiments

The experiments use ten scientific claims. The main routing comparisons and
depth study use publication dates from 2020-01-01 to 2026-10-30. Counts below
are deduplicated claim-paper pairs unless stated otherwise.

| Experiment | Main finding | Results |
|---|---|---|
| Original claim vs spaCy, top 10 | arXiv returned 10 vs 40 pairs; OpenAlex returned 100 with each method, with only nine shared | [Report](arxiv-concepts-20261031/report.md) |
| OpenAIRE discovery | Original and spaCy queries retrieved 112 pairs in total, including 94 not in the earlier OpenAlex/arXiv results | [Report](openaire-experiment-20261101/report.md) |
| spaCy vs KeyBERT, top 10 | OpenAlex had 47 vs 67 provisionally labelled evidence candidates. OpenAIRE's comparison remains incomplete because many spaCy results are unlabelled | [Comparison](keybert-experiment-20261101/final_strategy_comparison.md) |
| arXiv spaCy vs KeyBERT, top 10 | spaCy retrieved 40 pairs, KeyBERT 12; their union contained 42 | [Report](arxiv-keybert-20261101/report.md) |
| Depth and arXiv relaxation | Top 30 added 24 known provisional evidence candidates over top 10. Relaxation added 14 unlabelled candidates and reduced arXiv zero-result claims from six to two | [Report](readiness-live-results-20261101/report.md) |

## Current decisions

- Use KeyBERT for OpenAlex and OpenAIRE, and spaCy for arXiv.
- Retrieve up to 30 papers per provider, then deduplicate across providers.
- Keep arXiv relaxation off until its additional candidates are assessed.

The depth study fetched top 30 once per claim/provider and compared its 10/20/30
prefixes offline. These prefixes needed no extra requests and do not measure
separate request latencies. The depth run used 30 requests; the relaxation run
used 15. Both completed without API failures.

Labels are provisional AI-assisted judgments, not validated ground truth.
Unknown and conflicting labels are not counted as irrelevant. More candidates
do not establish better accuracy, and many deeper results remain unassessed.
OpenAIRE descriptions may not be abstracts; PDF links were not downloaded.

## Files to use

- [Detailed depth results and input hashes](readiness-live-results-20261101/results.json)
- [Depth run and response archive](depth-approved-run/)
- [Relaxation run and response archive](fallback-approved-run/)
- [14 relaxation additions for assessment](readiness-live-results-20261101/fallback_new_for_assessment.csv)
- [All 341 new or unresolved pairs](readiness-live-results-20261101/new_unresolved_for_assessment.csv)
- [12 conflicting labels](integration-readiness-20261101/conflicting_labels_12.csv)
- [47-example manual review sample](integration-readiness-20261101/manual_validation_47.csv)

Match labels by claim ID and canonical paper identity, not CSV row position.
Local model weights and runtime caches are ignored. Earlier reports may describe
previous query choices; use the comparisons above for the current design.
