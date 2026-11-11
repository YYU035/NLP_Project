# Comprehensive Person 3 Evaluation: Three Academic Datasets

Updated: 2026-11-09. Status: **complete for the declared Person 3 evaluation scope**.

The completed scope covers dataset/protocol validation, four-method paper ranking,
available gold sentence-evidence evaluation, component ablations, RRF sensitivity,
cached rerank-depth analysis, stage latency, uncertainty and result integrity.
It does not claim three-domain final-verdict Accuracy/Macro-F1 or expert evidence
sentence F1 on datasets that do not provide those labels.

## 1. Coverage and frozen protocols

| Dataset | Domain | Collection and evaluated units | Gold / scope |
|---|---|---|---|
| SciFact | Primarily biomedical | 5,183 abstracts; complete 300-claim dev run, 188 positive-eligible claims and 112 no-positive exclusions | Existing measured GPU ranking and sentence-rationale experiments |
| CSFCube | Computer science | 4,207 loaded papers; 24 fold1-dev and 26 fold1-test query/facet tasks | Expert graded per-query candidate pools; both official task groups evaluated |
| RECIPER | Materials science / EEE-related content | All 1,024 questions against the complete 343-paper collection | Single known source-paper ID per generated question; title/abstract retrieval |

The adapters pin source revisions/checksums and keep gold labels out of scoring
features. No new model training or parameter selection was performed on these
held-out task results. Primary RRF k=60 and Cross-Encoder prefix=20 were frozen.
RRF/Cross-Encoder experiments use real team module boundaries and provisioned
pretrained weights. Every new dataset ranking retains all declared candidates.

CSFCube dev/test query/facet IDs have zero overlap, but seven query-paper IDs
appear in both groups. The official split is not paper-disjoint. The two groups
are reported separately, without treating them as independent unseen-paper tests.
RECIPER is a frozen zero-shot evaluation of the released set, not a separately
held-out train/test study. Its generated source targets are not exhaustive expert
relevance judgments. Neither title/abstract adaptation is claimed to reproduce
all source papers' published model architectures or leaderboard scores.

## 2. Actual four-method ranking comparisons

Recall@10, MRR@10 and nDCG@10 are binary source/relevance metrics under each
dataset's stated gold. CSFCube also retains separate graded nDCG. Scores across
different tasks/candidate universes must not be pooled into an overall accuracy.

| Dataset / split | Method | Recall@10 | MRR@10 | Binary nDCG@10 |
|---|---|---:|---:|---:|
| CSFCube / fold1_dev | bm25 | 0.2230 | 0.4507 | 0.2802 |
| CSFCube / fold1_dev | sbert | 0.3898 | 0.6004 | 0.4324 |
| CSFCube / fold1_dev | rrf | 0.3348 | 0.5224 | 0.3856 |
| CSFCube / fold1_dev | rrf_cross_encoder | 0.3348 | 0.5936 | 0.3910 |
| CSFCube / fold1_test | bm25 | 0.1963 | 0.5510 | 0.2611 |
| CSFCube / fold1_test | sbert | 0.3733 | 0.6218 | 0.4273 |
| CSFCube / fold1_test | rrf | 0.3923 | 0.6482 | 0.4378 |
| CSFCube / fold1_test | rrf_cross_encoder | 0.3439 | 0.5255 | 0.3792 |
| RECIPER / entire_release_zero_shot | bm25 | 0.9033 | 0.7894 | 0.8172 |
| RECIPER / entire_release_zero_shot | sbert | 0.9053 | 0.7438 | 0.7831 |
| RECIPER / entire_release_zero_shot | rrf | 0.9365 | 0.8177 | 0.8467 |
| RECIPER / entire_release_zero_shot | rrf_cross_encoder | 0.9434 | 0.8302 | 0.8578 |
| SciFact / dev | bm25 | 0.9164 | 0.7844 | 0.8139 |
| SciFact / dev | sbert | 0.9117 | 0.7314 | 0.7743 |
| SciFact / dev | rrf | 0.9591 | 0.8164 | 0.8458 |
| SciFact / dev | rrf_cross_encoder | 0.9498 | 0.8561 | 0.8736 |

SciFact also includes actual Cross-Encoder depth-10 and depth-50 results in the
complete CSV. CSFCube supports a different conclusion from RECIPER: on its test
group, RRF has higher binary Recall@10/MRR@10 than the current Cross-Encoder
configuration. RECIPER improves in aggregate with fusion and again with reranking.
The report keeps these outcomes rather than retuning parameters on the test set.

## 3. Evidence correctness: SciFact gold sentence evaluation

The following are actual measured semantic selection results on 188 gold-eligible
claims, 366 gold-union sentences and 209 gold claim/document pairs. Sentence P/R/F1
uses exact source sentence IDs and the union of annotated rationales. Complete
rationale recovery separately accepts any fully recovered alternative rationale.
These are custom development diagnostics, not official SciFact leaderboard scores.

| Selection condition | Precision | Recall | Micro F1 | Complete-rationale document recall | No-gold selected sentences |
|---|---:|---:|---:|---:|---:|
| Oracle gold papers / sentence | 0.3748 | 0.5437 | 0.4437 | 0.7321 | 0 |
| RRF top-5 papers / sentence | 0.1093 | 0.4918 | 0.1788 | 0.6603 | 739 |
| RRF+CE20 top-5 papers / sentence | 0.1229 | 0.5109 | 0.1982 | 0.6794 | 711 |
| RRF top-5 papers / context | 0.0706 | 0.6776 | 0.1279 | 0.7703 | 1549 |

All 18 original selection conditions are exported in `evidence_summary.csv`.
Oracle-paper performance isolates selection; retrieved-paper conditions also
include missed or irrelevant documents. Evidence precision/F1 is weak even when
paper Recall is high. More context raises recall while introducing many extra
sentences, so source-linked extraction must not be described as solved.

The 112 no-gold claims are excluded from these positive-gold P/R/F1 denominators.
Their selected-sentence counts are reported explicitly rather than concealed or
assumed independently verified errors. The metrics therefore describe the stated
positive-gold diagnostic population, not all-claim official evidence accuracy.

CSFCube has paper grades and RECIPER has source-paper targets, without expert
sentence rationales. Evidence sentence F1 and final-verdict accuracy are marked
not applicable for those tasks; no labels or scores were invented.

## 4. Ablations and sensitivity

- Component comparison: BM25 only, SBERT only, both fused by RRF, and fusion plus
  Cross-Encoder are measured for every dataset.
- SciFact has actual Cross-Encoder prefix 10/20/50 comparisons.
- CSFCube dev/test and RECIPER include RRF k=20/60/100 sensitivity from exact
  exported lexical/dense rankings. Replayed k=60 exactly matches runtime order.
- Smaller CE prefixes 5/10 are derived by restricting the cached CE20 ordering
  to each original RRF prefix. All prefix candidates were scored, and the
  unchanged-tail constraint is checked. This is a fixed-score counterfactual
  replay, not fresh inference at each depth; no new latency savings are claimed.
- Depth 0 and 20 replay the actual baseline/CE20 exports. No unseen candidate
  scores or cross-products of new RRF constants and unscored CE pools are inferred.

All sensitivity settings are reported in `ablations.csv`; the primary configuration
remains frozen rather than selected from the largest test score.

## 5. Uncertainty and paired comparisons

Exploratory 95% percentile intervals use 2,000 cluster-bootstrap draws. CSFCube
resamples whole query-paper clusters (19 dev, 22 test); RECIPER resamples source
papers (343 clusters). Every repeated task within a sampled paper travels with
its cluster. Paired method differences use the same resampling weights.

| Dataset / split | MRR@10 difference | Estimate | 95% interval |
|---|---|---:|---|
| CSFCube / fold1_dev | rrf minus sbert | -0.0780 | [-0.1659, -0.0072] |
| CSFCube / fold1_dev | rrf_cross_encoder minus rrf | 0.0712 | [-0.0432, 0.1990] |
| CSFCube / fold1_test | rrf minus sbert | 0.0264 | [-0.1031, 0.1559] |
| CSFCube / fold1_test | rrf_cross_encoder minus rrf | -0.1227 | [-0.2899, 0.0444] |
| RECIPER / entire_release_zero_shot | rrf minus sbert | 0.0739 | [0.0540, 0.0925] |
| RECIPER / entire_release_zero_shot | rrf_cross_encoder minus rrf | 0.0126 | [-0.0037, 0.0293] |

These intervals describe variability under task-cluster resampling, not annotation
correctness, model retraining variation or arbitrary future queries. They are
exploratory multiple comparisons without multiplicity correction; no confirmatory
statistical-significance claim is made. Several differences have wide intervals.

SciFact's original full query-ranking artifacts are not in this checkout. Its
measured summaries and provenance are retained and their ranges/count arithmetic
are audited; query-level confidence intervals are not fabricated from aggregate
numbers. This does not replace the original GPU run with a claimed new run.

## 6. Latency and integrity

`latency.csv` reports actual per-stage means and nearest-rank p95 observations.
Model loading, dense indexing and repeated lexical indexing have separate scope
in the source reports. SciFact timings come from its original GPU development
run, CSFCube dev from CPU float64, and CSFCube test/RECIPER from local Apple MPS.
Hardware differs, so these numbers are not a cross-domain speed comparison or
a stable full-app throughput benchmark. Replayed ablations have no invented
inference-latency measurement.

The comprehensive manifest records checks for source versions, complete candidate
rankings, recomputed binary and graded metrics, exact RRF replay, CE-prefix
integrity, evidence micro/count arithmetic and split overlap. Original and
analysis outputs retain SHA-256 provenance. The dedicated RECIPER verification
also independently recomputes single-source metric formulas without the main
metrics implementation.

## 7. Reproduce and consume

Full comparison inputs are large runtime files, excluded from Git. Recreate them
with the CSFCube/RECIPER documented commands or use the archived originals. The
original SciFact summaries are committed. Then run:

```bash
PYTHONPATH=live_search/src HF_HUB_OFFLINE=1 \
.venv/bin/python -m ranking_evaluation.comprehensive_evaluation \
  --cs-dev outputs/datasets/csfcube-fold1-dev-four-methods/comparison.json \
  --cs-test outputs/datasets/csfcube-fold1-test-four-methods/comparison.json \
  --reciper outputs/datasets/reciper-full-343-papers-1024-questions-mps-four-methods/comparison.json \
  --bootstrap-repeats 2000 --output-dir outputs/datasets/new-comprehensive-report
```

Use a new output directory. Published compact artifacts are in
`ranking_evaluation/results/comprehensive-three-datasets/`: `ranking_summary.csv`,
`evidence_summary.csv`, `ablations.csv`, `uncertainty.csv`, `latency.csv` and
`manifest.json`. The app integration catalog links these files, so displaying
results does not require raw data downloads or model initialization.

Normal user claims still use live scholarly search. No frontend implementation,
XAI feature or final claim-verifier evaluation is claimed by this Person 3 handoff.
