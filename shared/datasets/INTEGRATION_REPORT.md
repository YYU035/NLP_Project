# Additional Dataset Integration Report

Date: 2026-11-09 (Asia/Shanghai). Branch: local `yuyue` after team synchronization.
Changes in this batch are uncommitted and unpushed.

## Implemented

- Version-pinned official downloads, file-size limits, SHA-256 validation and
  reuse of completed downloads. Data imports do not access the network.
- CSFCube corpus, query/facet, released-pool and adjudicated-grade loading;
  official fold selection; separate model inputs, judgments and ranking templates.
- CSFCube evaluation of supplied rankings with binary threshold metrics and
  separate linear-gain/log2 graded nDCG. Unknown/duplicate/out-of-pool IDs and
  pending/version-mismatched exports are rejected.
- A lexical pilot using Person 2's existing BM25 implementation, not a new
  retriever or manual ranking. No neural model downloaded for this work.
- Matter-of-Fact split loading, explicit EEE-related topic filtering, opaque
  case IDs and separate inference/gold/prediction-template files. Generated
  explanations, supporting facts and answer-bearing original IDs stay out of
  model-facing inputs.
- Binary feasibility prediction scoring with Accuracy, Macro-F1, abstention
  counts, complete-case checks and citation cutoff checks. This does not add
  or train a feasibility prediction model.

## Pinned releases and audited counts

| Dataset | Revision | Audited content |
|---|---|---|
| CSFCube | `7ffe012c1d92c470f185fc086086caf5b8e05ef3` | 4,207 paper records; 50 query/facet tasks; fold1 dev has 24 tasks |
| Matter-of-Fact | `a012d8182fbf25434c4389102e81ad282528a100` | 8,360 total cases; EEE subset: 936 train / 1,800 validation / 3,854 test |

EEE scope includes battery, semiconductor and superconductor themes; aerospace
records are retained in raw data but excluded from this declared subset.
Selected subsets are balanced between feasible/infeasible labels. No source
arXiv paper ID overlaps were found between release splits after version-suffix
normalization; this is not a semantic/model-training contamination guarantee.

Raw data is under ignored `data/csfcube/` and `data/matter_of_fact/` with manifests.
Complete audit: `outputs/datasets/dataset-audit-2026-11-09.json`.

## Actual CSFCube development pilot

One real BM25 run over 24 fold1-development query/facet tasks:

| Metric | Value |
|---|---:|
| MRR@10, relevant grade >= 2 | 0.4507 |
| Recall@10, relevant grade >= 2 | 0.2230 |
| Binary nDCG@10, relevant grade >= 2 | 0.2802 |
| Graded nDCG@10, linear gain | 0.4002 |

Protocol: full title/abstract plus explicit facet query; released judged pools;
query self-matches excluded; Person 2 BM25 k1=0.9/b=0.4, simple stemmer and
pool-order tie breaking. Two raw pools include self-matches with grade 3;
their exclusion is intentional and recorded, not an unreported source change.
No parameter selection was performed from these results. This is a development
pilot with supplemental metric definitions, not official scorer reproduction,
an open-corpus retrieval score, or a claim-verification result.

Report including input/evaluator/adapter hashes:
`outputs/datasets/csfcube-fold1-dev/bm25-evaluation-with-provenance.json`.

## Matter-of-Fact status

Real prepared EEE validation inputs: 1,800 cases at
`outputs/datasets/matter-of-fact-eee-validation/inference-input.json`.
The adjacent prediction template is pending. No actual model predictions,
accuracy or F1 results have been produced for this dataset by this batch.

The original task predicts feasibility with restricted historical evidence.
Binary labels do not automatically map to the system's three NLI verdicts.
Citation dates in submitted predictions are checked but not independently
authenticated; model-training exposure is not verified. Evaluator reports
therefore explicitly keep `formal_temporal_protocol_verified=false`.

## Validation and corrections

- New dataset tests: 19 cases covering source verification/failure cleanup,
  label separation, real BM25 calls, graded metric reference comparison,
  temporal/ID/version constraints, abstentions and evaluator CLIs/overwrite guards.
- Full ranking suite: **144 passed, 1 skipped**. The skipped pre-existing PDF
  test requires optional PyMuPDF; no skip is attributed to a new dataset test.
- The first CSFCube loader rejected raw self-matches; after checking real grade
  values, a documented self-exclusion protocol was added instead of silently
  accepting or modifying the release.
- An initial test incorrectly rejected the descriptive string
  `judgment="relevance_adju"` as label leakage. It was corrected to check actual
  document/query judgment keys; numeric gold remains separated.
- `git diff --check` and English-text checks passed. Runtime data/results remain
  ignored; no model download, API experiment, commit or push in this batch.

## Remaining work

1. Export real SBERT/RRF/Cross-Encoder rankings on the same CSFCube pools, with
   predeclared fold and graded relevance protocol.
2. Select an explicit feasibility inference policy and obtain temporally valid
   evidence for Matter-of-Fact; do not score pending or invented predictions.
3. No frontend dataset selector is required: the user confirmed evaluation-only
   datasets. The existing live paper-search UI remains unchanged.
4. Compare datasets by their actual tasks; do not report a single combined
   accuracy or claim three complete end-to-end verification benchmarks.
