# Matter-of-Fact EEE: Historical Retrieval and Evidence Pilot

> Archived exploratory experiment; this dataset is not currently selected as the third primary benchmark.

Date: 2026-11-09. Status: real search/ranking/evidence execution completed;
feasibility inference and independent evidence correctness evaluation pending.

## What actually ran

The runner consumed only prepared claim text, an opaque join ID and the
per-claim knowledge cutoff. It selected the first 10 validation cases sorted by
opaque case ID, without looking at labels, source-paper IDs or generated facts.
This is a small operational pilot, not the complete 1,800-case validation run.

1. Person 1 `RetrievalService`: live OpenAlex and OpenAIRE calls, maximum 10
   papers per provider, historical end-date filter, actual responses archived.
2. Additional date checks reject candidates with unknown/future dates in either
   the merged paper or any retained source alternative.
3. Person 2 BM25 and SBERT score the returned candidate pool for each claim.
4. Person 3 RRF k=60 and Cross-Encoder top 20 rank that pool.
5. Person 3 semantic selection examines the top five papers, using the existing
   SciFact-train-frozen threshold 0.5, at most three sentences per paper, no
   context expansion. It extracts abstracts only, with exact source offsets.

No PDF downloads, NLI inference or feasibility predictions occurred. Missing
abstracts, below-threshold papers and zero-result cases remain explicit.

## Measured operational results

| Quantity | Result |
|---|---:|
| Attempted / processed claims | 10 / 10 |
| Claims with dated candidates | 8 / 10 |
| Candidate paper occurrences | 70 |
| Unique candidate identities in this pilot | 70 |
| Top-ranked papers examined for evidence | 38 |
| Claims with selected passages | 5 / 10 |
| Selected exact abstract passages | 12 |
| Rejected future/unknown-date candidate records | 0 |
| Mean search time per attempted claim | 1.9582 seconds |
| Mean post-search processing time per attempted claim | 0.9997 seconds |

The two empty searches were valid zero-result responses, not inferred negative
feasibility labels. OpenAlex returned papers for eight claims and zero results
for two. OpenAIRE returned zero results for all ten. No provider failure was
reported. Dates are provider metadata, not independently authenticated version
histories. Zero date rejections does not prove that no temporal leakage exists.

Timing covers one synchronous CPU run. Post-search timing includes BM25/SBERT
indexing, retrieval scoring, fusion, reranking and evidence selection. These
means exclude model loading, setup and output serialization. They are not GPU
throughput or full application latency. Candidate counts are not Recall@K;
passage counts/coverage are not evidence precision or correctness.

## Reproducibility

Models:

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- CPU float64 model execution was explicitly enabled due to the previously
  observed local CPU float32 nonfinite-score issue.

The query planner reuses Person 1's KeyBERT phrase generator with the provisioned
ranking encoder. This is an explicit experimental encoder override, not Person
1's default pinned query encoder. arXiv was not enabled in this pilot because
its spaCy query model was not provisioned. The normal application configuration
and Person 1 source files were not modified.

From the repository root, install direct dependencies if necessary:

```bash
.venv/bin/python -m pip install -r ranking_evaluation/requirements-datasets.txt
```

Then run (requires public scholarly API access and provisioned local models):

```bash
PYTHONPATH=live_search/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python -m ranking_evaluation.run_matter_retrieval \
  --input outputs/datasets/matter-of-fact-eee-validation/inference-input.json \
  --output-dir outputs/datasets/matter-eee-historical-pilot-10 \
  --limit 10 --top-k 10 --paper-depth 5 \
  --encoder-path .cache/ranking/semantic-MiniLM-L6-v2 \
  --encoder-id sentence-transformers/all-MiniLM-L6-v2 \
  --encoder-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --reranker-path .cache/ranking/ms-marco-MiniLM-L6-v2 \
  --reranker-id cross-encoder/ms-marco-MiniLM-L6-v2 \
  --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
  --cpu-float64
```

Use a new output directory for another run. Search results can change; use the
archived responses and per-case discoveries to inspect this specific run.
The summary includes model-file hashes, package versions, input hash and runner
hash. Each case has `discovery.json` and `pipeline.json`, including full retained
source metadata, ranking IDs/scores, original claim and exact selected spans.

Artifacts: `outputs/datasets/matter-eee-historical-pilot-10/` (Git-ignored).
A separate one-claim preliminary connectivity probe exists at
`outputs/datasets/matter-historical-probe/`; it is excluded from the counts above.
`human-review.json` in the ten-case directory contains 38 paper review records
with blank independent relevance/sufficiency labels. It has not been reviewed
and must not be treated as gold or used to claim manual evaluation results.

## What this does and does not establish

The actual team search/retrieval/ranking/evidence modules can process this
materials-domain claim sample with historical metadata filtering and preserved
sources. Half of the attempted claims yield selected passages with the frozen
selection threshold. No threshold was tuned on these ten claims.

There is no released human sentence-evidence gold or exhaustive paper qrels for
this task. Consequently this run does not establish paper relevance, evidence
correctness or final feasibility Accuracy/Macro-F1. In particular, absence of
retrieved support does not mean a scientific claim is infeasible. Three-way NLI
labels must not be silently substituted for these feasibility labels. This is
also not yet a Person 4 handoff: passages have not passed its tokenizer budget.

Keep reporting SciFact's gold paper/evidence metrics for the instructor's second
requirement, CSFCube's graded paper relevance metrics, and this operational
materials pilot separately. The claim-only baseline in
[MATTER_BASELINE_REPORT.md](MATTER_BASELINE_REPORT.md) is a different experiment;
its 64.22% Accuracy is not the accuracy of this retrieval pipeline.

Next required scientific step: specify and validate a feasibility predictor on
training data with historical sources, then score independent development/test
predictions. If the intended requirement is the same gold retrieval/evidence
metrics across all three datasets, choose an appropriately annotated third
benchmark or add independently validated annotations before claiming completion.

## Verification

Ranking test suite after adding this runner: 152 passed, 1 skipped. Focused
integration tests use real BM25/fusion/reranking/evidence code with injected
models to check empty pools, date rejection, claim preservation, exact source
substrings and gold-field rejection. The live run above separately exercises
actual encoders, reranking weights and scholarly APIs. `pip check` and
`git diff --check` passed. No frontend changes or Git push were performed.
