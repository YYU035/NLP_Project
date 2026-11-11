# Regression investigation — 11 November 2026

The primary policy and frozen audit are unchanged. This report explains all 21 regressions using recorded native predictions; **zero new model calls and zero threshold/model changes**.

## Findings

- **10/21: paper removal is sufficient to lose focused correctness.** Reaggregating the original predictions using only papers retained by the new candidate changes a correct verdict into an incorrect one. Context can also contribute; this does not establish the gate as the sole cause.
- **11/21: retained-context predictions are needed for the changed result.** Reaggregating focused predictions on the candidate paper scope remains correct; the actual context-based predictions change the result.
- **Retrieval order is identical in 21/21.** No retrieval change explains these paired regressions. Retrieval quality can still limit both versions.
- 10 regressions lose at least one previously accepted annotated-gold paper. 1 have no annotated gold paper in either top-five scope. 5 positive cases had a correct baseline label without selecting any complete annotated gold sentence.

| Benchmark → primary verdict | Cases |
|---|---:|
| CONTRADICTED → CONFLICTING | 1 |
| CONTRADICTED → INSUFFICIENT | 8 |
| CONTRADICTED → SUPPORTED | 2 |
| INSUFFICIENT → SUPPORTED | 4 |
| SUPPORTED → INSUFFICIENT | 6 |

Four no-positive cases become Supported. These are benchmark errors, but absent annotations do not establish that the retrieved evidence is scientifically wrong; human review must inspect the sources.

## Review first

Open **index.html** for the 9-case guide, then fill **REVIEW_FORM.csv**. The first eight are deterministic diagnostic claim cases; the last is a real PDF source check. All 21 regressions have complete source pages and baseline/primary replay JSONs. No human ratings are filled in.

## Responsibilities

| Owner | Check | Status |
|---|---|---|
| Yu Yue | Inspect gate-discarded annotated papers and whether whole-abstract context hides qualifying sentences; propose experiments only on development data. | PENDING human review |
| Chong Yao | Inspect retained-paper NLI changes, especially Contradicted → Supported/Conflicting, and confidence despite wrong benchmark outcomes. | PENDING human review |
| Naila | Inspect sentence relevance and explain model sensitivity on development examples; rankings did not change in these regressions. | PENDING human review |
| Raghavi | Test fresh discovery/provenance/provider failures separately; this offline regression comparison does not evaluate web recall. | PENDING live testing |
| Kireeti | Compare exact source text, displayed scope, quotations, export/replay and the PDF pages. Automated source/replay checks pass; semantic ratings remain blank. | PENDING human review |

## Interpretation

No confirmed frontend correctness defect was found in these traces: recorded source spans match the corpus and replay/grounding checks pass. This does not validate the scientific meaning. These losses arise under the configured gate/context policy; do not patch or tune that policy using this inspected audit.

The guard experiment is retained as a comparison, not selectively substituted for failed claims. Overall audit remains 45 fixes /21 regressions; no improved accuracy is claimed from this diagnostic report.
