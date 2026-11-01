# Original versus spaCy concept retrieval: provisional assessment

Read-only analysis of the saved 31 October 2026 experiment, prepared 1 November 2026. **No API calls, retrieval changes or new experiments.**

## Inputs and counting

- [Provisional labels](manual_assessment_provisional_250.csv), [original paper export](manual_assessment.csv), [saved results](results.json), [request log](calls.jsonl), and the referenced `responses/*.body` files.
- Verified all 250 labelled-file positions and their unchanged claim, query, paper metadata and identity against the original export; recomputed the existing conservative DOI/arXiv-ID/title grouping from saved results. Matched labels by **(claim_id, identity)**, not title or row order.
- **250 returned-record instances; 234 unique claim?paper pairs; 226 labelled and 8 unlabelled.** No conflicting labels among duplicate pairs. The 234-pair distribution is 15 label-0, 85 label-1, 126 label-2 and 8 unknown. Decimal CSV values `0.0/1.0/2.0` were interpreted as 0/1/2.
- Labels are **provisional AI-assisted judgments, not validated ground truth**. Relevant means labels 1 or 2; evidence means label 2 (a subset of relevant). Unknowns are reported separately and are not treated as irrelevant. Counts below are known-labelled counts, not final relevance estimates.
- Both arms used 2020-01-01 through 2026-10-30, one request per claim/provider/arm, and **top_k=10**. There were 40 saved calls with no failures. arXiv uses submission dates. This is the saved **backend concept ablation**, not a new run of the generator modes.

## Provider-level comparison

Unique counts are within each provider/arm across the ten claims; source identities are not summed across overlapping arms.

| Provider | Method | Total records | Unique pairs | Relevant (1+2) | Evidence (2) | Unlabelled |
|---|---|---:|---:|---:|---:|---:|
| openalex | Original | 100 | 100 | 90 | 54 | 4 |
| openalex | spaCy concepts | 100 | 100 | 94 | 51 | 4 |
| arxiv | Original | 10 | 10 | 10 | 10 | 0 |
| arxiv | spaCy concepts | 40 | 40 | 33 | 25 | 0 |

Concept-only additions exclude identities already returned by the original query **for the same claim and provider**. Displaced papers are original-only identities absent from the concept arm; these are different from additions.

| Provider | Unique overlap | New relevant / evidence from concepts | Original-only relevant / evidence | Net known relevant / evidence change | Two-arm unique union |
|---|---:|---:|---:|---:|---:|
| openalex | 9 | 85 / 43 | 81 / 46 | +4 / -3 | 191 |
| arxiv | 5 | 28 / 20 | 5 / 5 | +23 / +15 | 45 |

OpenAlex has four unlabelled pairs in each arm (all exclusive to that arm). Its apparent +4 topical / -3 evidence difference is therefore incomplete. The OpenAlex 191-pair and arXiv 45-pair unions overlap across providers; the combined union is **234**, not 236. A union retains both result lists and costs two requests per claim/provider, versus one in either matched arm.

## Original and concept queries

The original column is the exact canonical scientific claim. OpenAlex searched it as text; the arXiv control rendered its words as a strict `all:... AND all:...` expression. The exact complete wire expressions and date filters are in `calls.jsonl` under `attempts[].params`. The concept columns below show the saved provider-specific query text: OpenAlex used space-separated phrases, while arXiv used quoted phrase constraints. No query has been regenerated for this report.

| Claim | Original canonical query | OpenAlex concept query | arXiv concept expression (before date filter) |
|---|---|---|---|
| c01 | The results of this study show that the expression of TNF-α does not increase the risk of COVID-19 in older adults. | `TNF-α COVID-19` | `all:"TNF-α" AND all:"COVID-19"` |
| c02 | In this study, we demonstrate that lithium-ion batteries retain 90% capacity after 500 cycles at -5 °C. | `lithium-ion batteries capacity` | `all:"lithium-ion batteries" AND all:capacity` |
| c03 | Our experiments show that the efficiency of SiC MOSFETs is not lower than that of silicon devices at 100 kHz. | `SiC MOSFETs silicon devices` | `all:"SiC MOSFETs" AND all:"silicon devices"` |
| c04 | This study demonstrates that elevated atmospheric CO2 concentrations increase ocean acidification in coastal waters. | `elevated atmospheric CO2 concentrations ocean acidification` | `all:"elevated atmospheric CO2 concentrations" AND all:"ocean acidification"` |
| c05 | The presence of dark matter increases the rotational velocity of spiral galaxies. | `dark matter rotational velocity` | `all:"dark matter" AND all:"rotational velocity"` |
| c06 | Experimental results show that deficit irrigation reduces water use without decreasing wheat yield under drought conditions. | `deficit irrigation water use` | `all:"deficit irrigation" AND all:"water use"` |
| c07 | Preliminary findings suggest that vitamin D supplementation may reduce respiratory infections in older adults. | `vitamin D supplementation respiratory infections` | `all:"vitamin D supplementation" AND all:"respiratory infections"` |
| c08 | Our results show that graph neural networks improve molecular property prediction compared with random forests. | `graph neural networks molecular property prediction` | `all:"graph neural networks" AND all:"molecular property prediction"` |
| c09 | It has been demonstrated that platinum nanoparticles accelerate hydrogen evolution at 25 °C in acidic solutions. | `platinum nanoparticles hydrogen evolution` | `all:"platinum nanoparticles" AND all:"hydrogen evolution"` |
| c10 | Exercise improves memory. | `Exercise memory` | `all:Exercise AND all:memory` |

## Per-claim, per-provider results

`T/U` = returned total / unique pairs. `R/E` = known relevant / known evidence pairs. **? = top-10 capped**: every nonempty response returned exactly 10 and its archived API total exceeded 10. These are confirmed truncations, not evidence that only ten papers exist. `0` means a valid empty response. Unknown labels are shown as original/concepts.

| Claim | Provider | Original T/U | Original R/E | Concepts T/U | Concepts R/E | Unknown O/C |
|---|---|---:|---:|---:|---:|---:|
| c01 | openalex | 10/10? | 7/2 | 10/10? | 10/3 | 1/0 |
| c01 | arxiv | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| c02 | openalex | 10/10? | 4/1 | 10/10? | 5/0 | 2/3 |
| c02 | arxiv | 0/0 | 0/0 | 10/10? | 3/0 | 0/0 |
| c03 | openalex | 10/10? | 10/5 | 10/10? | 10/5 | 0/0 |
| c03 | arxiv | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| c04 | openalex | 10/10? | 10/7 | 10/10? | 10/7 | 0/0 |
| c04 | arxiv | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| c05 | openalex | 10/10? | 10/8 | 10/10? | 10/5 | 0/0 |
| c05 | arxiv | 0/0 | 0/0 | 10/10? | 10/9 | 0/0 |
| c06 | openalex | 10/10? | 10/7 | 10/10? | 9/0 | 0/1 |
| c06 | arxiv | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| c07 | openalex | 10/10? | 9/4 | 10/10? | 10/9 | 1/0 |
| c07 | arxiv | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| c08 | openalex | 10/10? | 10/8 | 10/10? | 10/6 | 0/0 |
| c08 | arxiv | 0/0 | 0/0 | 10/10? | 10/8 | 0/0 |
| c09 | openalex | 10/10? | 10/6 | 10/10? | 10/9 | 0/0 |
| c09 | arxiv | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| c10 | openalex | 10/10? | 10/6 | 10/10? | 10/7 | 0/0 |
| c10 | arxiv | 10/10? | 10/10 | 10/10? | 10/8 | 0/0 |

## Low retrieval: observations versus explanations

- **OpenAlex:** every original and concept query is capped at ten. There are **no queries returning fewer than five**. Its archived totals range from 311 to 387,781 matches; these totals describe index matches, not relevant papers.
- **arXiv original:** zero for c01?c09. Only c10 returns candidates: ten saved from 22 reported matches.
- **arXiv concepts:** zero for **c01 (TNF-?/COVID-19), c03 (SiC/silicon devices), c04 (CO2/acidification), c06 (irrigation/wheat), c07 (vitamin D/infections), c09 (platinum/hydrogen evolution)**. The other four are capped: c02 207 reported matches, c05 104, c08 154, c10 115. There are **no nonzero lists containing one to four papers**.

| Cases | Confirmed observation | Interpretation / remaining uncertainty |
|---|---|---|
| arXiv c02, c05, c08 | Changing query construction with the same provider/date window changes 0 returned papers to 10, with API totals above 10. | Direct evidence that relaxing the original construction recovers candidates. It does not isolate which removed words caused the zeros or establish that all additional candidates are relevant. |
| arXiv c01, c03, c04, c06, c07, c09 | Both arms have HTTP-successful empty responses and reported total 0. | No parsing/transport failure is evident. Exact-phrase/conjunction restrictions, term variants, the date window and corpus coverage remain competing hypotheses. This run cannot isolate them. The older [provider diagnosis](../../docs/provider_diagnostics.md) also found c01 literal TNF-?/COVID-19 empty without dates, making dates alone an inadequate explanation for that specific expression. |
| arXiv c04 | Saved concepts retain the long exact phrase `elevated atmospheric CO2 concentrations`. | Potentially restrictive wording is observable; whether a shorter phrase would recover relevant candidates is untested. Do not label corpus coverage as the proven cause. |
| OpenAlex c06 | Concept query drops wheat yield/drought context; known evidence falls 7 to 0 (one concept result is unlabelled). | The omission and the result change are confirmed. That omission caused the loss is a plausible explanation, not an isolated causal test. |
| arXiv c02 | Concepts return ten, but only three are provisionally topical and none label 2. | Candidate volume alone is insufficient: broad battery/capacity matching appears weak for the specific retention/temperature claim under these labels. |

## Recommendation

1. **Retain spaCy concepts as a complementary discovery option, not a universal replacement.** arXiv gains known relevant candidates (10 to 33) and evidence candidates (10 to 25); OpenAlex is mixed (relevant 90 to 94, evidence 54 to 51, with four unknowns per arm). This supports practical usefulness, not validated superiority.
2. **Keep original queries available.** OpenAlex concepts improve known evidence for c01, c07, c09 and c10, tie c03/c04, but lose it for c02/c05/c06/c08. arXiv c10 also has a net fall of two label-2 candidates in the matched comparison. Concept-only additions do not cancel the importance of original-only evidence.
3. **Insufficient candidates:** the six arXiv concept zeros above remain unresolved; arXiv originals are insufficient for c01?c09. Every claim still has ten OpenAlex candidates per arm, so there is no cross-provider total-count shortage. However, c02 has only one known OpenAlex original evidence paper and none from either concept provider; c06 has no known concept evidence. These are evidence-candidate weaknesses, not merely list-size issues.
4. **Proceed with downstream integration while retaining both query paths and provenance.** Before relying on a concept-only policy, review the eight unknowns and a sample of AI-assisted labels. A targeted future refinement should preserve critical outcomes/populations such as wheat yield and conditionally relax long phrases; do not restore every-word conjunction.
5. **Increase limits only for capped queries when more candidates are needed.** Pagination or a larger future cap could expose more candidates for the 25 capped responses, but relevance is untested beyond ten. Raising top_k cannot fix the 15 saved zero-result responses because their reported total is zero. No limit change or new search was performed here.

All findings depend on a small ten-claim sample, provisional labels, top-10 truncation and one live snapshot. No statistical significance, corpus recall, verified PDF access or claim truth is asserted.
