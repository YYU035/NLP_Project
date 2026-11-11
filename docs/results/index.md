# Results and validation index

These studies predate directory consolidation. Migration validation is separate from scientific accuracy; moving files is not a new benchmark.

## Current primary policy

Read the [209-claim internal audit](../../evaluation/independent_results_20261111/REPORT.md), [per-claim results](../../evaluation/independent_results_20261111/per_claim.csv) and [changed claims](../../evaluation/independent_results_20261111/changed_claims.csv).

| Metric | Original baseline | Bounded-context primary |
|---|---:|---:|
| Accuracy | 56.5% | 67.9% |
| Macro F1 | 0.583 | 0.679 |
| Contradiction precision | 47.8% | 84.0% |
| Contradiction recall | 82.1% | 53.8% |
| Input evidence F1 | 0.210 | 0.165 |

The 209 reserved published SciFact training claims have cited/evidence-paper IDs disjoint from logged prior training and inspected development work. Ranking uses all 5,183 corpus papers and frozen settings. The candidate fixes 45 errors and introduces 21 regressions with zero processing failures. Accuracy gain is 11.5 percentage points, with cited-paper-cluster 95% interval +3.8 to +18.7 points.

This is **an internal audit, not an official hidden test or external-domain proof**. Pretraining exposure and related uncited statements cannot be ruled out. Further tuning requires separate development data.

## Three-domain retrieval

Read the [full validation report](../../evaluation/validation/FINAL_TEST_REPORT.md) for denominators, controls and protocols.

| Dataset / scope | RRF Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| SciFact dev: 188 positive queries, 5,183 papers | 95.6% | 0.806 | 0.833 |
| CSFCube fold1 test: 26 query-facet tasks, judged pools | 39.2% | 0.648 | 0.438 |
| RECIPER: 1,024 questions, 343 titles/abstracts | 93.7% | 0.818 | 0.847 |

Protocols and encoder/preprocessing settings differ. CSFCube test is small and shares some query-paper identities with development. RECIPER has a single source-paper target rather than exhaustive relevance. Only SciFact supplies the verdict/rationale gold used here. This table is not pooled verification accuracy or public-web recall.

## Earlier research and retained failures

- [300-claim baseline](../../evaluation/validation/FINAL_TEST_REPORT.md): 56.7% accuracy and 0.584 macro F1 on previously inspected SciFact dev; oracle controls remain separate.
- [Context study](../../evaluation/context_study/REPORT.md): exploratory policies on that 300-claim cohort; 66.0% accuracy / 79.7% contradiction recall is a different comparison from the 209-claim audit.
- [All 21 audit regressions](../../evaluation/regression_review_20261111/ALL_REGRESSIONS.md) and [offline review pack](../../evaluation/regression_review_20261111/index.html): complete traced failures, recoveries and blank human forms.
- [Baseline error catalogue](../../evaluation/validation/results/error_catalogue.csv): retained full-route/synthetic failures with overlapping diagnostic tags.

Historical reports retain original commands, paths and test counts. Use current root commands and the migration manifest for the consolidated source. Frozen baselines must not silently be rewritten to match new code.

## Reliability and explanation quality

Software passes establish bounded functional checks, not scientific accuracy. Earlier checks include seven PDF cases, 30 repeated retrieval lists, five repeated NLI pairs and 20 primary-policy subset changes without extra neural forwards. Synthetic language expectations passed 12/15; population/generalisation errors remain.

The four-pair word-occlusion pilot is exploratory. IG real-case numerical reliability remains unresolved. All 12 recorded raw local-Qwen generations failed schema validation and fell back to templates. The prepared explanation-review study has zero completed ratings; mechanically valid templates are not human usefulness scores.

## Consolidation checks

[Migration validation](migration_validation.md) records independent installation, software checks, preserved evidence and bounded real-model smoke checks.
