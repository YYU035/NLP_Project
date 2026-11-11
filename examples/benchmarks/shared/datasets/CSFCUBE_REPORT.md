# CSFCube: Computer-Science Paper Ranking

Date: 2026-11-09. Active evaluation set: SciFact, CSFCube and RECIPER.

## Measured protocol

This experiment uses all 24 released fold1-development query/facet tasks and
4,207 loaded paper records. Each task ranks its released judged candidate pool;
it is not full-corpus live search or scientific-claim verification. The two folds
exchange development/test assignments; this result is development evaluation.

Official source: https://github.com/iesl/CSFCube. Pinned revision:
`7ffe012c1d92c470f185fc086086caf5b8e05ef3` (CC BY-NC 4.0). The download manifest
contains SHA-256 for the exact files. Query text combines the explicit facet name
with the unchanged query-paper title and abstract. Query-paper self matches are
excluded from candidates and gold; source files are preserved unchanged.

Binary positives are adjudicated grades >=2. Supplemental graded nDCG preserves
grades 0–3 with linear gain and log2 discount. The metric aggregation is declared
explicitly and is not claimed identical to the official scorer. No-positive
queries are excluded with counts rather than assigned an invented success.

Actual P2 BM25/SBERT and P3 RRF/Cross-Encoder were used. BM25 is k1=0.9, b=0.4,
simple stemmer; RRF k=60; Cross-Encoder top 20. This run used CPU float64 model
weights due to the previously observed local CPU nonfinite-score issue.

## Actual fold1-development results

| Method | Recall@10 | MRR@10 | Binary nDCG@10 | Graded nDCG@10 |
|---|---:|---:|---:|---:|
| bm25 | 0.223019 | 0.450694 | 0.280197 | 0.400245 |
| sbert | 0.389752 | 0.600397 | 0.432392 | 0.534896 |
| rrf | 0.334824 | 0.522388 | 0.385636 | 0.501981 |
| rrf_cross_encoder | 0.334794 | 0.593568 | 0.390991 | 0.504990 |

SBERT outperformed the current hybrid configurations on these tasks. Do not
claim that RRF or Cross-Encoder always improves every dataset. No parameters
were selected or retuned from these results. Sentence-evidence and verdict
labels are not supplied by this collection.

Compact release artifacts: `ranking_evaluation/results/csfcube-fold1-dev/summary.csv`
and `manifest.json`. The latter retains the measured model/data/code provenance
and the hash of the full local `comparison.json`. Full predictions remain at
`outputs/datasets/csfcube-fold1-dev-four-methods/`. Recorded source hashes describe
that run; the shared runner later gained additive MPS synchronization, source
metadata and progress callbacks without changing this declared scoring protocol.

## Reproduce

Provision local encoder/reranker weights explicitly as described in the dataset
and model setup instructions. Use the same model IDs/revisions as the RECIPER
report and a fresh output directory:

```bash
.venv/bin/python -m shared.datasets.setup csfcube
PYTHONPATH=live_search/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python -m ranking_evaluation.run_csfcube \
  --split fold1_dev --facet all \
  --encoder-path .cache/ranking/semantic-MiniLM-L6-v2 \
  --encoder-id sentence-transformers/all-MiniLM-L6-v2 \
  --encoder-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --reranker-path .cache/ranking/ms-marco-MiniLM-L6-v2 \
  --reranker-id cross-encoder/ms-marco-MiniLM-L6-v2 \
  --reranker-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
  --device cpu --cpu-float64 \
  --output-dir outputs/datasets/new-csfcube-four-methods
```

Use `--device cuda` without `--cpu-float64` for an independently recorded GPU
run. Per-query scores/rankings and stage timings are saved by the runner.
Changing hardware/dtype changes the timing conditions; a single development
run is not a stable full-application throughput benchmark.

## Completed official fold1 test group

The same frozen models, RRF k=60 and Cross-Encoder depth=20 were evaluated on
all 26 official fold1-test query/facet tasks. No parameter was selected from
these labels. The run used local MPS float32 weights and four Torch CPU threads.
Dev/test task IDs are disjoint, while seven query-paper IDs overlap. This is
the official task split, not a paper-disjoint generalization experiment.

| Method | Recall@10 | MRR@10 | Binary nDCG@10 | Graded nDCG@10 |
|---|---:|---:|---:|---:|
| bm25 | 0.196336 | 0.551007 | 0.261070 | 0.378822 |
| sbert | 0.373317 | 0.621795 | 0.427253 | 0.572423 |
| rrf | 0.392318 | 0.648184 | 0.437848 | 0.565380 |
| rrf_cross_encoder | 0.343892 | 0.525519 | 0.379227 | 0.518662 |

RRF has the strongest binary Recall@10/MRR@10 point estimates here. The current
Cross-Encoder reduces those averages; the complete results are retained. No
claim of uniform reranking improvement is made. CPU development and MPS test
timings are different execution conditions and are not pooled.

Compact test artifacts: `ranking_evaluation/results/csfcube-fold1-test/`.
The app catalog now uses this test summary by default and retains development
paths separately. For the test command above, use `--split fold1_test`,
`--device mps --threads 4` without `--cpu-float64`, and a fresh output directory.
Cluster intervals, paired comparisons and replay ablations are in the
[comprehensive report](../COMPREHENSIVE_EVALUATION.md).
