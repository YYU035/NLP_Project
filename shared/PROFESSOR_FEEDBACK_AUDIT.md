# Professor Feedback: Evidence Audit

> **Current selection update:** The active set is SciFact + CSFCube + RECIPER. The user selected the materials candidate after file-level screening. See [RECIPER protocol/results](datasets/RECIPER_REPORT.md). Earlier Matter-of-Fact and ClimateCheck selections/results below are historical.

Updated: 2026-11-09 (Asia/Shanghai). This audit reads saved results; it does not rerun model inference or assert instructor acceptance.

## Requirement 1: at least three datasets across domains

**Status: three data sources/adapters are present; three completed model evaluations are not yet available.**

| Dataset | Domain | Actual evaluation evidence | Remaining gap |
|---|---|---|---|
| SciFact | Primarily biomedical science | Existing complete dev ranking/evidence summaries from the merged Person 3 branch | Preserve development qualification; maintain reproducibility and held-out evaluation claims accurately |
| CSFCube | Computer science | New 24-task fold1-dev four-method ranking run | Held-out-fold evaluation and final protocol/report; no sentence-evidence labels |
| Matter-of-Fact | EEE-related materials | Official data audit, prepared inputs and a prediction evaluator | Actual temporally qualified feasibility predictions and measured scores |

Download counts or passing parser tests alone do not constitute evaluation. These datasets also measure different tasks; do not claim three identical end-to-end scientific-verification experiments.

## Requirement 2: retrieve relevant papers and identify correct evidence

**Status: the required metric families are implemented and real SciFact results exist for both layers. A defensible report must show the measured strengths and weaknesses, not only list metric names.**

### A. Paper relevance/ranking — SciFact development

5,183 corpus documents; 300 claims processed. Relevance macro metrics use 188 gold-evidence claims; 112 no-gold claims are excluded with counts. Results below were saved in the merged branch and were not rerun during this audit.

| Method | CE depth | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| bm25 | - | 0.9164 | 0.7844 | 0.8139 |
| sbert | - | 0.9117 | 0.7314 | 0.7743 |
| rrf | - | 0.9591 | 0.8164 | 0.8458 |
| rrf_cross_encoder | 10 | 0.9591 | 0.8518 | 0.8749 |
| rrf_cross_encoder | 20 | 0.9498 | 0.8561 | 0.8736 |
| rrf_cross_encoder | 50 | 0.9605 | 0.8550 | 0.8761 |

This demonstrates that known relevant papers can be recovered on this fixed development corpus. It does not establish all-domain live API recall. Reranking can improve early ranking while slightly lowering Recall at a specific cutoff. Timings are also saved but are one-run stage observations, not stable end-to-end performance.

### B. Correct evidence selection — SciFact development

Exact source sentence-ID matching, micro Precision/Recall/F1, over 188 gold-evidence claims. Alternative rationales are retained; the union diagnostic is not the official SciFact scorer. No-gold claims are excluded from P/R/F1 with their predictions counted separately.

| Selector | Paper input | P | R | F1 | Complete-rationale document recall |
|---|---|---:|---:|---:|---:|
| semantic_sentence | oracle_gold_papers (oracle diagnostic) | 0.3748 | 0.5437 | 0.4437 | 0.7321 |
| semantic_sentence | rrf top 5 | 0.1093 | 0.4918 | 0.1788 | 0.6603 |
| semantic_sentence | rrf_ce_top20 top 5 | 0.1229 | 0.5109 | 0.1982 | 0.6794 |
| semantic_context | rrf top 5 | 0.0706 | 0.6776 | 0.1279 | 0.7703 |

**Quality limitation:** evidence precision is low for retrieved-paper inputs. For RRF+CE top-20 then top-5 papers, precision is about 12.3%, recall about 51.1%, and micro F1 about 19.8%. This supports a measured limitation, not a claim that evidence is reliably correct. Oracle-paper scores must not be presented as end-to-end results. Context increases annotated-sentence coverage while selecting more non-rationale text. Incomplete annotations can penalize useful text, but do not by themselves explain away the measured error rate.

Existing failure analysis distinguishes missing gold papers, irrelevant selected passages, incomplete rationales and missing context. See [the saved checklist](../ranking_evaluation/results/person3-v2/CHECKLIST_REPORT.md). Manual PDF review is still pending; an AI-reviewed convenience sample is not independent human ground truth.

### C. Additional-domain ranking — CSFCube development

Same 24 development query/facet tasks and released candidate pools, explicit self-match exclusion; binary relevance is grade >=2. Graded nDCG retains linear gains 0..3. This is paper-pool ranking, not claim verification.

| Method | Recall@10 | MRR@10 | Graded nDCG@10 |
|---|---:|---:|---:|
| bm25 | 0.2230 | 0.4507 | 0.4002 |
| sbert | 0.3898 | 0.6004 | 0.5349 |
| rrf | 0.3348 | 0.5224 | 0.5020 |
| rrf_cross_encoder | 0.3348 | 0.5936 | 0.5050 |

SBERT is stronger than the current hybrid configuration on this pilot. Do not claim universal RRF improvement or silently select another configuration on the test fold.

## Minimum remaining work to close the feedback honestly

1. Complete an actual Matter-of-Fact feasibility run with a declared evidence/time protocol, then score predictions. Parser/template availability is insufficient. If its original task cannot be supported within project constraints, replace it rather than invent a result.
2. Present paper metrics and evidence metrics as separate result tables, including baselines, sample sizes, exclusions, settings and domain/task qualifications. Existing SciFact summaries already provide the core numerical evidence for requirement 2.
3. Fix protocols before held-out evaluation; distinguish train-selected settings, development results and untouched final testing. Add the relevant error analysis and limitations.
4. Do not equate unit-test counts, model smoke checks, gold-paper diagnostics or a few downloaded PDFs with scientific quality metrics.
5. No frontend changes or forced three-benchmark runtime routing are needed. Live search remains independent of evaluation corpora.

The instructor may require stronger results or more cross-domain evidence metrics; this audit cannot guarantee grading. The quoted feedback does not itself require every metric on every dataset, and missing gold annotations must not be fabricated.

## Source artifacts

- [SciFact paper metrics](../ranking_evaluation/results/person3-v2/retrieval-results.csv)
- [SciFact evidence metrics](../ranking_evaluation/results/person3-v2/semantic-dev-summary.json)
- [CSFCube four-method results](../outputs/datasets/csfcube-fold1-dev-four-methods/summary.csv)
- [Dataset integration status](datasets/EVALUATION_STATUS.md)

## Artifact fingerprints

- `ranking_evaluation/results/person3-v2/retrieval-results.csv`: SHA-256 `d9f828644faf116064f8f69b180c9fe2e4d907c3ce13def9413f1d89bd1dc5dc`
- `ranking_evaluation/results/person3-v2/semantic-dev-summary.json`: SHA-256 `dd503fcd4a6d10cb07355350413bc91c0f1dc27e73f1a48a7dd8e2b5810b17e6`
- `outputs/datasets/csfcube-fold1-dev-four-methods/summary.csv`: SHA-256 `654c720e04ab4ca35966755a553821d2cefa44f858d2218b5a664f98bdf85b60`

Dataset changes and this audit remain local, uncommitted and unpushed. No teammate approval or independent rerun was assumed.

## Update: third-dataset diagnostic baseline (2026-11-09)

Matter-of-Fact now has a real claim-only TF-IDF/logistic-regression validation baseline: 936 training claims, 1,800 validation claims, Accuracy 0.642222 and Macro-F1 0.640678. See [the experiment report](datasets/MATTER_BASELINE_REPORT.md). Earlier statements about missing all Matter-of-Fact model results are superseded by this baseline. The project system has **not** been evaluated on this dataset: this classifier measures neither retrieval relevance nor evidence correctness. Three measured datasets must not be described as three completed system evaluations.

## Update: real historical retrieval/evidence pilot (2026-11-09)

The [materials pilot](datasets/MATTER_HISTORICAL_PILOT.md) now executes real Person 1 search, Person 2 retrieval, and Person 3 hybrid ranking/evidence selection: 10 claims, 70 candidates, 12 selected abstract passages across 5 claims. It does not provide final feasibility predictions or gold evidence-correctness scores. The earlier no-live-run status is superseded; the full-system evaluation gap remains explicit.

## Active replacement measured: ClimateCheck (2026-11-09)

ClimateCheck now replaces Matter-of-Fact in the primary three-dataset evaluation. Four actual team ranking configurations were run on all 176 exact test claims with human-judged abstract pools; 166 positive-eligible claims contribute to macro ranking metrics and 10 all-NEI claims are excluded explicitly. RRF+CE Recall@2 is 0.254937 and MRR@10 is 0.770453; SBERT has higher MRR@10 (0.795841). See [the report](datasets/CLIMATECHECK_REPORT.md) and the compact CSV/manifest under `ranking_evaluation/results/climatecheck-judged-pool/`. This is a real gold ranking evaluation, narrower than full-corpus retrieval. ClimateCheck has abstract-level labels, not sentence rationale gold. Actual verdict metrics still require inference, and no claim of three completed end-to-end evaluations is made.

## Selected third dataset completed: RECIPER (2026-11-09)

The active set is now SciFact + CSFCube + RECIPER. A real four-method MPS run
completed all 1,024 questions against the same full 343-paper title/abstract
collection, without gold candidate preselection or parameter tuning. RRF+CE
achieved source-paper Recall@10 0.943359 and MRR@10 0.830227. Independent
single-source metric arithmetic, corpus completeness and reranking-prefix checks
passed. The ranking suite has 162 passes and one pre-existing optional PDF skip.
See [the measured report](datasets/RECIPER_REPORT.md),
[team handoff](EVALUATION_HANDOFF.md) and compact results under
`ranking_evaluation/results/reciper-title-abstract/`.

The three datasets now have actual task-specific retrieval/ranking results.
This does not mean three completed end-to-end verification evaluations: CSFCube
has paper grades; RECIPER has generated-QA source targets, without expert
sentence/verdict labels. SciFact supplies the actual sentence-evidence P/R/F1
results, whose weak precision remains a material limitation. Final verdict
Accuracy/Macro-F1 and instructor acceptance remain separate from this result.

## Comprehensive Person 3 evaluation completed

See [COMPREHENSIVE_EVALUATION.md](COMPREHENSIVE_EVALUATION.md) for the final task-specific report. The missing 26-task CSFCube fold1 test experiment is now measured. The report consolidates all three datasets, four-method comparisons, component/sensitivity replay ablations, stage timings, exploratory cluster confidence intervals, existing SciFact evidence diagnostics and integrity audits. Person 4 final-verdict evaluation and XAI/frontend work remain separate; no unsupported labels or metrics are claimed.
