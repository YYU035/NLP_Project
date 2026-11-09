# ClimateCheck Replacement: Protocol and Measured Ranking Results

> Archived exploratory experiment; this dataset is not currently selected as the third primary benchmark.

Date: 2026-11-09. Active datasets: SciFact, CSFCube, ClimateCheck.

## Selection decision

ClimateCheck replaces Matter-of-Fact in the primary evaluation set. Matter-of-Fact
is a materials feasibility benchmark with time-dependent knowledge restrictions,
not a ready gold paper-ranking/evidence-selection benchmark. Its previously run
claim-only classifier cannot measure the project's retrieval quality. Keep its
code and reports as archived exploration, not primary evidence for the rubric.

ClimateCheck asks systems to retrieve scientific publications for climate claims
and classify each claim–abstract pair as Supports, Refutes or Not Enough
Information. The authors describe double annotation by climate/environmental
science students with curator resolution. This fits the intended scientific
paper/evidence workflow while adding an environmental-science domain to the
biomedical and computer-science datasets. Claims originate from public discourse;
this remains a distribution difference from student research-assistance queries.
It is not a pure EEE or physics benchmark.

Primary sources:

- Authors' code/data description: https://github.com/ryabhmd/climatecheck
- Dataset and license: https://huggingface.co/datasets/rabuahmad/climatecheck
- Shared-task overview: https://aclanthology.org/2025.sdp-1.24/
- Separate full corpus: https://huggingface.co/datasets/rabuahmad/climatecheck_publications_corpus

## Audited release and data integrity

Pinned pair-data revision: `93d0dc5007e9f86a3fc4cfe2d7998514a94f5c88`.
The download manifest retains SHA-256 for the dataset card and both Parquet files.
The card declares MIT; original scientific text attribution/rights still apply.

| Released split | Annotated pairs | Distinct numeric claim IDs | Exact claim texts/tasks | Abstract IDs |
|---|---:|---:|---:|---:|
| Train | 3,023 | 763 | 782 | 2,382 |
| Test | 1,904 | 172 | 176 | 1,685 |

Test labels: 749 Supports, 266 Refutes, 889 Not Enough Information.
All test rows have data_version 2025. Current training data mixes 2025/2026 rows;
this experiment did not train or tune on either split. Do not silently present
all current training rows as the original 2025 release.

Four numeric test claim IDs have two different texts each. The adapter uses the
numeric ID plus SHA-256 of the unchanged claim text, yielding 176 tasks. For
example, two different entities sharing an upstream numeric ID must not have
their claims or relevance judgments merged. The adapter rejects duplicate exact
claim–abstract pairs and conflicting texts for the same abstract ID.

## Actual experiment protocol

This first experiment ranks **only each claim's released human-judged abstracts**.
It does not retrieve from the complete 394,269-abstract publication corpus.
The pair files are small and do not contain that full corpus. A pool assembled
from gold annotation membership is a closed reranking test, not an open search
experiment. Do not compare its numbers with the full-corpus shared-task leaderboard.

- Query: unchanged released claim text. No gold label, narrative annotation or
  explanatory metadata becomes a query feature.
- Documents: unchanged abstracts; titles are unavailable in the pair files and
  are explicitly empty, rather than inferred from evidence labels.
- Positive relevance: Supports **or** Refutes. Either can be useful evidence.
  NEI means non-evidentiary under the annotation, not necessarily off-topic.
- Full candidate pool is scored before stable ordering; tie order is the source
  pool order. Every judged candidate is retained in each method's full ranking.
- P2 BM25: k1=0.9, b=0.4, simple stemmer; P2 SBERT: cosine-normalized embeddings.
- P3 fusion: RRF k=60; optional Cross-Encoder reranks its top 20. Pools have at
  most 17 members, so this reranks all available candidates for every test claim.
- Encoder input length is capped by the provisioned model (256 tokens); the
  Cross-Encoder uses 512 tokens for claim/document pairs. Long abstracts may be
  truncated for scoring; the full source abstract is retained in output.
- Supplemental metrics: Recall/MRR/binary nDCG at 1, 2, 5, 10, plus untruncated
  MRR. Queries without positive gold are excluded from these macro means and
  their count is reported. No official Bpref/scorer parity is claimed.
- No parameter search or threshold selection was performed on these test labels.
- Encoder: `sentence-transformers/all-MiniLM-L6-v2` at
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L6-v2` at
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Local CPU float64 execution explicitly enabled due to the earlier local CPU
  nonfinite-score issue. Stage timing is a single run, not GPU throughput.

## Measured results: human-judged candidate pools

All 176 exact test claim tasks were ranked. The macro metrics below include
166 claims with at least one Supports/Refutes abstract; 10 all-NEI claims are
excluded and reported explicitly. Candidate-pool membership is supplied by
annotations, so these scores do not measure discovery among 394,269 papers.

| Method | Recall@2 | MRR@10 | Binary nDCG@2 | Recall@10 |
|---|---:|---:|---:|---:|
| bm25 | 0.198564 | 0.707913 | 0.518971 | 0.870020 |
| sbert | 0.249313 | 0.795841 | 0.642180 | 0.899504 |
| rrf | 0.231295 | 0.733400 | 0.576768 | 0.891831 |
| rrf_cross_encoder | 0.254937 | 0.770453 | 0.627223 | 0.900344 |

RRF plus Cross-Encoder has the best Recall@2 in this experiment, while SBERT
has better MRR@10 and nDCG@2. Fusion/reranking is not uniformly better. These
are point estimates from one run, not a statistical-significance claim. No
configuration was selected or retuned using these results.

Actual SBERT document indexing took 52.2506 seconds on this local CPU run.
Per-query component timings are retained in the full JSON.


## Reproduce

Models must already be provisioned locally. Setup is explicit and imports do
not download data/models:

```bash
.venv/bin/python -m pip install -r ranking_evaluation/requirements-datasets.txt
.venv/bin/python -m shared.datasets.setup climatecheck

PYTHONPATH=live_search/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python -m ranking_evaluation.run_climatecheck \
  --split test \
  --encoder-path .cache/ranking/semantic-MiniLM-L6-v2 \
  --encoder-id sentence-transformers/all-MiniLM-L6-v2 \
  --encoder-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --reranker-path .cache/ranking/ms-marco-MiniLM-L6-v2 \
  --reranker-id cross-encoder/ms-marco-MiniLM-L6-v2 \
  --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
  --cpu-float64 \
  --output-dir outputs/datasets/climatecheck-test-judged-pools-four-methods
```

Use a fresh output path for another run. Output `comparison.json` includes every
method's full rankings, per-query metrics, stage timings, data protocol, model-file
hashes and code hashes. `summary.csv` provides the small comparison table. Raw
data and full runtime outputs remain Git-ignored. A compact reviewable CSV and
manifest are retained at `ranking_evaluation/results/climatecheck-judged-pool/`.
This report preserves the protocol and measured summary for review.

## What remains

ClimateCheck has **abstract-level** gold, not sentence-level evidence spans.
Use it for evidentiary abstract ranking and actual three-way abstract verification
when Person 4 inference is run. SciFact continues to measure sentence-evidence
precision/recall/F1. Do not claim three-domain sentence-evidence evaluation.

Full-corpus retrieval requires downloading/indexing the separate publication
corpus and accounting for incomplete human judgments. The observed corpus
revision is `ace5f2a1c6418f56630d999b01e5e06bce828dba`; its Parquet file is
894,098,148 bytes. It has not been downloaded in this replacement batch.
Provision it on the remote host for the full retrieval experiment, preserve
abstract IDs and do not treat all unjudged documents as known negatives.

Actual Person 4 predictions and Accuracy/Macro-F1/weighted F1 are still pending.
No feasibility-to-NLI label conversion is needed for this dataset's declared
three-way abstract task. No frontend or Person 1 API changes are needed for
fixed-dataset evaluation.

## Verification and Git state

156 ranking tests passed, 1 pre-existing optional PDF test skipped. New checks
cover conflicting upstream claim IDs, duplicate pairs, abstract consistency,
complete pool membership, hand-calculated MRR/Recall and no-positive exclusions.
Existing CSFCube integration tests also passed after the ranking runner was made
reusable. These tests verify implementation; scientific numbers come from the
actual model run above. Changes remain local, uncommitted and unpushed.
