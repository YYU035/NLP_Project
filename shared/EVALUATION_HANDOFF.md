# Person 3 Evaluation Handoff

Updated: 2026-11-09. Active evaluation datasets: SciFact, CSFCube and RECIPER.

The [comprehensive evaluation report](COMPREHENSIVE_EVALUATION.md) is the primary
report for completed task coverage, all comparisons, ablations, latency, cluster
uncertainty and evidence diagnostics.

## What was added

Dataset preparation, pinned revisions/checksums and shared adapters are under
`shared/datasets/`. Benchmark runners, scoring code and tests are under
`ranking_evaluation/`. Experiments call Person 2's actual BM25/SBERT modules and
Person 3's RRF/Cross-Encoder boundary. Raw corpora, model weights and large
prediction files remain outside Git; compact measured results and provenance
are retained with English experiment reports.

| Dataset | Domain | Actual result scope | Gold boundary |
|---|---|---|---|
| SciFact | Primarily biomedical | Existing full-development paper ranking and source-linked sentence-evidence experiments in the merged branch | Paper labels and sentence rationales; evidence quality remains limited |
| CSFCube | Computer science | Completed four-method comparisons on 24 fold1-dev and 26 fold1-test tasks; test is the default result view | Expert graded paper-pool relevance; no verdict/sentence gold |
| RECIPER | Materials science, including EEE-related research | Completed all 1,024 questions against all 343 papers; RRF+CE Recall@10 0.943359, MRR@10 0.830227 | 1,024 generated questions with source-paper targets across all 343 papers; no expert sentence/verdict gold |

RECIPER replaces the earlier exploratory third-dataset choices. Matter-of-Fact
and ClimateCheck code/reports are retained for history but excluded from the
active three-dataset comparison.

## Metrics and reporting

Paper ranking: Recall@K, MRR, nDCG and measured stage latency. RECIPER's single
source target makes Recall@K equivalent to source-paper Hit@K; it is not an
exhaustive relevance judgment of all other papers. Its title/abstract baseline
also differs from the original RECIPER paragraph/recipe experiment.

Sentence evidence: SciFact precision, recall, F1 and complete-rationale recovery.
Keep oracle-paper and retrieved-paper conditions separate. The merged results
show weak retrieved-evidence precision/F1, so do not describe evidence selection
as solved or uniformly reliable. CSFCube/RECIPER cannot produce expert evidence
sentence F1 without additional independent annotations.

Final verdicts: Accuracy/Macro-F1 requires actual compatible verification
inference. A successful ranking run or unit-test suite does not establish final
scientific-verdict accuracy. Do not combine incompatible metrics across these
three tasks into an overall system accuracy.

## Integration boundary

These datasets are fixed evaluation resources. Normal claims still use Person
1's live scholarly APIs; no mandatory three-dataset search router or frontend
selector was added. Frontend work remains outside this change.

Person 3 owns benchmark orchestration, fusion/reranking and result reporting,
using Person 2's retrievers. Person 4's model and evidence/label contract are
needed for its own final-verdict experiment. RECIPER questions must not be
silently converted into artificial scientific-verdict labels.

## Where to review

- [RECIPER experiment and reproducible command](datasets/RECIPER_REPORT.md)
- [Dataset setup and protocols](datasets/README.md)
- [Evaluation status and remaining boundaries](datasets/EVALUATION_STATUS.md)
- Existing SciFact results: `ranking_evaluation/results/person3-v2/`
- CSFCube full local predictions: `outputs/datasets/csfcube-fold1-dev-four-methods/`
- RECIPER full local predictions:
  `outputs/datasets/reciper-full-343-papers-1024-questions-mps-four-methods/`
- Compact RECIPER CSV/manifest after completion:
  `ranking_evaluation/results/reciper-title-abstract/`

## Validation

165 ranking tests passed; one pre-existing optional PDF test skipped. New checks
cover source-label separation, valid source references, complete corpus rankings,
hand-calculated metrics, finite neural scoring and GPU synchronization. A
separate verifier recomputes single-source metrics directly from saved rankings
without using the main metric functions and checks the reranking-prefix boundary.
Its full-run checks passed and are recorded alongside the measured RECIPER results.

Code, reports and experiments are English. Raw corpora, model weights and large outputs are provisioned separately from Git.
