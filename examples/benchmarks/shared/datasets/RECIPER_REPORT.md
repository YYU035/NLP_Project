# RECIPER: Materials Source-Paper Retrieval Evaluation

Date: 2026-11-09. Active benchmarks: SciFact, CSFCube and RECIPER.

## Task and dataset

RECIPER was selected after inspecting actual data, source labels and annotation
limitations. It adds materials-science research, including battery/electronic
materials content, to SciFact's biomedical and CSFCube's computer-science scope.
It is not a complete EEE curriculum or a purely semiconductor collection.

Official resources:

- https://github.com/ReaganWu/RECIPER
- https://huggingface.co/datasets/ReaganWZY/RECIPER
- https://arxiv.org/html/2604.11229v1

Pinned release: `1e167badbf19d6fd78b2a49700028f4c68134219`. Setup downloads
only the dataset card, license, corpus JSON and QA JSONL. The manifest retains
SHA-256 for each file. All 343 corpus papers have title and abstract text. All
1,024 unique questions have a valid single source-paper target; none is missing.
The corpus also contains 12,162 paragraph entries and 544 generated recipes,
but those are not indexed in this first title/abstract experiment.

The questions/answers were generated using an LLM. Source-paper IDs provide
known provenance targets, not exhaustive expert relevance judgments. Other
papers may also answer a question. The release has no expert sentence/paragraph
rationale IDs or Supported/Contradicted/NEI labels. Do not report this benchmark
as evidence sentence F1 or scientific-verdict accuracy.

## Frozen experiment protocol

- Evaluate **all 1,024 released questions against all 343 papers**. No per-query
  pool is selected using gold targets; every method returns a complete paper ranking.
- This is a zero-shot evaluation of the entire released set. No training, parameter
  search or development/test model selection was performed on RECIPER labels.
- Query: exact original question only. Generated answers, topics, source IDs and
  gold paper titles are not appended to queries or provided as scoring features.
- Documents: original title plus original abstract. Generated recipes and full
  text are excluded. This matches Person 2's current paper representation.
- Source labels are kept in a separate evaluator-only export. The model-facing
  export contains only document text/metadata, query text and the shared candidate IDs.
- Actual Person 2 BM25: k1=0.9, b=0.4, simple stemmer; lexical index includes the
  complete corpus. SBERT: normalized cosine scores for the complete corpus.
- Actual Person 3 RRF: k=60. Optional Cross-Encoder reorders only the first 20
  fused candidates; the remaining 323 candidates retain their fused order.
- Stable lexical/dense ties use corpus order; Cross-Encoder ties use incoming
  RRF order. No candidate or query is silently removed.
- Metrics: source-paper Recall@1/5/10/20, MRR (full ranking and at each K), binary
  nDCG@K. With one target per question, Recall@K is also source-paper Hit@K.
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, local model input limit 256 tokens.
- Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`, maximum pair length 512, batch 16.
- Main run: local Apple GPU (`mps`), float32 model weights, four Torch CPU
  threads. SBERT retains Person 2's normalized float32 vector scoring implementation.
  Long texts may be truncated for neural scoring; original corpus text is retained.

This is an adapted **paper-level title/abstract retrieval** experiment. The
original RECIPER paper indexes paragraph/recipe streams and then collapses hits
to paper IDs. Therefore the published reference scores are not directly comparable
to this project's scores, and are not copied into its result table.

## Measured results

Completed all 1,024 questions against the complete 343-paper collection.
All four rankings contain every paper for every question; no query was excluded.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Binary nDCG@10 |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.727539 | 0.871094 | 0.903320 | 0.789400 | 0.817162 |
| sbert | 0.657227 | 0.851562 | 0.905273 | 0.743812 | 0.783101 |
| rrf | 0.753906 | 0.907227 | 0.936523 | 0.817665 | 0.846697 |
| rrf_cross_encoder | 0.767578 | 0.908203 | 0.943359 | 0.830227 | 0.857826 |

RRF improved source recovery over either single retriever in this run. Adding
Cross-Encoder reranking further improved aggregate metrics. This differs from
CSFCube, where the current hybrid configuration did not consistently beat SBERT.
Do not claim that fusion or reranking always improves every dataset or question.

The Cross-Encoder improved the known source rank for 133 questions, worsened it
for 113 and left it unchanged for 778. It recovered source top-1 for 79 cases
and lost source top-1 for 65. For 38 questions the source lay outside the RRF
prefix of 20 and therefore could not be promoted by this reranking stage.
These are known-source rank effects, not expert judgments that every other
paper is scientifically irrelevant. Nine fixed-order cases are retained for review.

Actual dense indexing: 3.4686 seconds. Measured elapsed time
through inference, before output serialization: 948.80 seconds.
This was one shared-workstation MPS run, not an isolated warmed-throughput benchmark.

Independent single-source arithmetic checks passed for all reported metric
cutoffs and all methods. Complete 343-paper rankings and the top-20-only
reordering constraint also passed. See the compact CSV, manifest, verification
report and source-rank examples in `ranking_evaluation/results/reciper-title-abstract/`.


## Timing policy

The report stores initial SBERT indexing time, model load time, lexical index
construction per query, BM25/SBERT query scoring, RRF and Cross-Encoder timings.
This implementation rebuilds the small lexical index per query to share the
existing fixed-pool runner; that overhead is measured separately, not hidden.
Model loading, initial dense indexing and JSON output are not folded into each
query's ranking-stage latency. These are single-run stage timings, not warmed
throughput or complete application latency. CUDA and MPS runs explicitly synchronize
through the scorer/runner boundaries; the main local run uses MPS.

## Reproduce

From the repository root, with the local model directories already provisioned:

```bash
.venv/bin/python -m pip install -r ranking_evaluation/requirements-datasets.txt
.venv/bin/python -m shared.datasets.setup reciper

PYTHONPATH=live_search/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python -m ranking_evaluation.run_reciper \
  --encoder-path .cache/ranking/semantic-MiniLM-L6-v2 \
  --encoder-id sentence-transformers/all-MiniLM-L6-v2 \
  --encoder-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --reranker-path .cache/ranking/ms-marco-MiniLM-L6-v2 \
  --reranker-id cross-encoder/ms-marco-MiniLM-L6-v2 \
  --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
  --device mps --threads 4 \
  --output-dir outputs/datasets/reciper-full-343-papers-1024-questions-mps-four-methods
```

Use a new output directory for another run. For a remote NVIDIA GPU, select
`--device cuda`. For local CPU fallback, use `--device cpu --cpu-float64`. Record
the changed device/dtype and environment rather than assuming equivalent timing.

Files in that output directory:

- `status.json`: actual completed query count and running/complete/failed status.
- `inputs/retrieval-input.json`: model-facing collection and questions.
- `inputs/judgments.json`: separate evaluator-only source-paper targets.
- `comparison.json`: all four complete rankings, per-query gold metrics, timings,
  model/data/code hashes and the actual package/hardware environment.
- `summary.csv`: method-level metric comparison.

A preliminary CPU float64 run was deliberately interrupted before completion
(last reported progress: 96/1,024) after a label-free hardware probe verified
finite model outputs and faster MPS execution. Its output directory
`outputs/datasets/reciper-full-343-papers-1024-questions-four-methods/` is marked
interrupted; it has no completed metric table and is excluded from the results.
The full MPS run starts from question one. No CPU/GPU scores are mixed.

Raw data, weights and large runtime outputs remain outside Git. A compact result
CSV and provenance manifest are retained under
`ranking_evaluation/results/reciper-title-abstract/` for the completed run.
Imports do not download models/data; network access is only used by explicit setup.

## Interpretation and remaining scope

This completes a new domain's measured source-paper retrieval comparison. It does not establish evidence passage correctness or final verdict
accuracy. SciFact remains the gold sentence-evidence benchmark. CSFCube remains
a graded expert paper-ranking benchmark. Present separate task-specific tables;
do not average their scores into an overall system accuracy.

Matter-of-Fact and ClimateCheck are retained as archived experiments and are
excluded from the active three-dataset result set. Person 1's live search and
the frontend are unchanged. Final Person 4 verification requires actual inference
on a suitably labeled benchmark and remains distinct from these ranking results.

## Checks and current state

162 ranking tests passed; one pre-existing optional PDF test skipped. The actual
MPS model run completed all four methods. Independent metric/export checks passed.
All 1,024 QA source titles also exactly matched their referenced corpus titles.
These checks do not certify generated-answer correctness or expert relevance.
The repository handoff includes code, compact measured results and English reports; raw data and weights remain outside Git.
