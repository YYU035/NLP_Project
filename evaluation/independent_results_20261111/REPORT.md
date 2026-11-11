# Frozen-candidate internal holdout audit — 11 November 2026

**209 previously unused claims**, with cited/evidence paper IDs disjoint from the logged selection-training and inspected development claims. All eligible claims were retained. This is an internal audit from the published SciFact training release, not an official hidden test or a new external dataset.

## Frozen comparison

Same full 5,183-paper corpus, production MiniLM retrieval, Porter BM25, RRF 60/top 100 branches/top 5 papers, pretrained NLI weights and 0.5 NLI threshold. Candidate: exact bounded abstract context plus frozen 0.65 paper gate. No settings chosen or changed from these results.

| Metric | Original baseline | Yu Yue frozen candidate |
|---|---:|---:|
| Accuracy | 56.5% | 67.9% |
| Macro F1 | 0.583 | 0.679 |
| SUPPORTED precision | 83.0% | 81.3% |
| SUPPORTED recall | 40.6% | 63.5% |
| CONTRADICTED precision | 47.8% | 84.0% |
| CONTRADICTED recall | 82.1% | 53.8% |
| INSUFFICIENT precision | 56.6% | 56.6% |
| INSUFFICIENT recall | 63.5% | 81.1% |
| Input evidence F1 | 0.210 | 0.165 |

Paired accuracy change: +11.5%; 95% cited-paper-cluster bootstrap interval [+3.8%, +18.7%], 149 connected citation clusters. Claim-level interval is retained separately; clustering is the primary assessment to reflect related claims.

Fixes: 45; regressions: 21; operational failures: 0. All 418 condition rows are retained. All 2010 cached NLI predictions have finite valid class probabilities, no truncation and operational status OK. Exact source/budget checks and grounded template gates ran during inference.

## Decision

Eligible for primary abstract-provider promotion under the predeclared overall-accuracy rule; class tradeoffs remain explicit.

The comparison assesses the abstract verification route. Public-web recall, human explanation usefulness, PDF extraction order and external-domain verdict accuracy are separate questions. CSFCube/RECIPER retrieval scores do not establish verdict accuracy. The no-positive label is closed-benchmark INSUFFICIENT, not a universal claim about literature.

## Reproducibility and limitations

The frozen cohort excludes prior SHA-selected 200 training IDs, conservatively also the first 200 raw training IDs, every inspected dev claim text (normalized), and all cited/evidence paper IDs of those claims. It does not prove no past teammate ever inspected any remaining published training record; pretrained-model exposure cannot be ruled out. Related uncited scientific statements can remain.

Protocol, complete per-claim outcomes and every changed verdict are in this folder. Local exact model-input traces and real-model replay examples are under outputs/independent_audit_20261111. Models/corpora remain local and ignored. No gold papers were injected into retrieval.

Reproduce ranking/cohort once with `python -m explanation_frontend.evaluation.independent_audit prepare`; the protocol refuses overwrite. Run shards 0/1/2 with `... independent_audit run --shard N`, then `python -m explanation_frontend.evaluation.independent_report`. Existing exact-input model predictions can be reused after interruption. No threshold search is allowed on this cohort.

## Application activation

The primary abstract provider is now enabled for new Live/Selected search jobs and the default pipeline_smoke command. PDF selection and direct LocalPipeline baseline behaviour are unchanged. Older replays retain their original metadata/results.

Software suite: 177 passed; dependency check passed. Actual-model subset test: 20 changes with 0 extra neural forwards; replay roundtrip passed. Fresh live primary smoke status: ok; its claim is not independently adjudicated.

Use EXPLANATION_ABSTRACT_POLICY=baseline for the original app provider; pipeline_smoke --baseline retains the original CLI route. Primary exports include frozen context configuration and source commit.
