# Three-Dataset Evaluation Status

Updated: 2026-11-09. Active selection: **SciFact, CSFCube, RECIPER**.

| Dataset | Actual evaluation scope | Boundary |
|---|---|---|
| SciFact | Existing full-dev GPU paper ranking and sentence-evidence selection results | Low evidence precision; final verification metrics are a separate experiment |
| CSFCube | Measured 24-task fold1-dev BM25/SBERT/RRF/RRF+CE comparison | Expert graded pool ranking; no claim-verdict or sentence-rationale gold |
| RECIPER | Completed four-method evaluation: all 1,024 questions over all 343 papers; RRF+CE Recall@10 0.943359, MRR@10 0.830227 | See RECIPER_REPORT.md for measured run status/results; generated QA, no sentence/verdict gold |

[RECIPER protocol/results](RECIPER_REPORT.md),
[CSFCube protocol/results](CSFCUBE_REPORT.md),
[team integration handoff](../EVALUATION_HANDOFF.md).

## What the metrics establish

Paper ranking: Recall@K, MRR, nDCG and measured latency. RECIPER metrics use the
known single source paper and do not establish exhaustive relevance of other
papers. Every question faces the full fixed collection, with no gold pool
preselection. Its original paragraph/recipe benchmark differs from this adapted
title/abstract paper-retrieval experiment.

Correct evidence: SciFact supplies actual gold paper labels and sentence
rationales, permitting evidence precision/recall/F1 and rationale recovery.
RECIPER supplies paragraphs but no expert gold passage/sentence IDs; CSFCube
supplies paper grades. Do not invent evidence F1 on those two collections or
claim three-domain sentence-quality validation.

Final verdict: Accuracy and Macro-F1 require actual Person 4 inference on a
labeled compatible task. Ranking experiments alone do not establish final
verification correctness or completed three-domain end-to-end evaluation.

## Runtime and ownership

Fixed evaluation corpora do not restrict Person 1's live API searches. P3 invokes
P2 retrieval and P3 ranking modules; a runtime dataset router or frontend change
is not needed. Person 4's checkpoint and evidence/label contract are needed for
its own final-verdict experiment. No frontend files were changed.

## Archived exploratory datasets

Matter-of-Fact's claim-only feasibility baseline and 10-claim historical
search/evidence pilot remain available. They lacked the required ready relevance
gold and are excluded from the active three-dataset table.
ClimateCheck has measured gold judged-pool ranking results, but its climate
subject domain was declined by the user and it is also excluded.
See the individual archived reports for exact numbers and limitations.

Raw data/models and large outputs remain Git-ignored. Compact result summaries,
English protocols and code form the repository handoff.
