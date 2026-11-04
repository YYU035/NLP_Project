# Hybrid Ranking and Evaluation

Person 3 implementation: RRF fusion, optional Cross-Encoder reranking, retrieval metrics, fixed-corpus evaluation and experiment grids. Uses Person 1's existing paper schemas. Actual Person 2 rankings and full team integration are still pending; synthetic examples do not establish retrieval quality.

## Setup and tests

Run from the repository root with Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'pydantic>=2.7,<3' 'pytest>=8,<10'
PYTHONPATH=live_search/src .venv/bin/python -m pytest ranking_evaluation/tests -q
```

The lightweight path only imports Person 1's schemas from `live_search/src`; live search itself requires its own setup. Optional NumPy/scikit-learn reference checks skip when those packages are absent. Python 3.13.1 was tested locally; other Python versions and GPU hosts remain unverified.

## Application interface

```python
from ranking_evaluation.schemas import RankingInput
from ranking_evaluation.pipeline import RankingConfig, RankingOutput, rank_claim

inputs = RankingInput(
    claim_id="claim-1",
    candidates=person1_result,  # Person 1 RetrievalResult, preserving metadata.
    bm25_ranking=bm25_ids,     # Most relevant first, using entry.identity.
    sbert_ranking=sbert_ids,
    bm25_seconds=None,        # Unknown timings stay None.
    sbert_seconds=None,
)
result = rank_claim(inputs, config=RankingConfig(output_top_k=10))
restored = RankingOutput.model_validate_json(result.model_dump_json())
```

Rankings must contain unique known identities. Resolve aliases to preferred IDs before passing lists; preserve identity aliases and source references downstream. The original claim must match every candidate. Input remains unchanged.

Output includes complete inputs/diagnostics, final `papers`, configuration, model description, union/scored counts, warnings and timings. Each ranked entry retains its upstream paper, one-based final rank, RRF score, branch ranks and optional model score. Empty results do not imply a verification verdict. PDF links are not verified full text; evidence extraction belongs to Person 4.

## Fusion and reranking

`rrf.reciprocal_rank_fusion` sums `1 / (rrf_k + branch_rank)` over present branches. Ranks start at 1; absent branches contribute zero. Default `rrf_k=60`; exact ties use ascending identity. Only ranked candidates enter the union. Output truncation follows fusion and, in `rank_claim`, optional reranking.

For real reranking, explicitly install and download:

```sh
.venv/bin/python -m pip install -r ranking_evaluation/requirements-rerank.txt
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m ranking_evaluation.scripts.setup_reranker
```

The setup script pins `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a` and records file hashes under ignored `.cache/`. The adapter requires local files, never downloads implicitly, and defaults to CPU.

```python
from ranking_evaluation.reranker import LocalCrossEncoder
from ranking_evaluation.pipeline import ModelDescription

scorer = LocalCrossEncoder(local_model_directory, device="cpu", batch_size=16)
model = ModelDescription(
    model_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    device="cpu", dtype="default", max_length=512, batch_size=16,
)
result = rank_claim(
    inputs, config=RankingConfig(rerank_enabled=True, rerank_top_n=20, output_top_k=10),
    scorer=scorer, model=model,
)
```

Reuse the loaded scorer across claims. Reranking uses original claim plus title/abstract, joining available fields with a blank line. Only the first N entries participate; missing-text papers keep their slots, ties preserve incoming order, and the tail stays unchanged. Model failures, non-finite or multi-class scores raise explicitly. Scores are relevance values, not probabilities or verification labels.

Known local issue: default CPU precision returned NaN in two tested dependency combinations; explicit `cpu_float64=True` passed a small real-model smoke check. The root cause remains unresolved. If enabling this option, set `ModelDescription.dtype="float64"` too. Do not assume it is needed or validated on other hosts. Scientific-domain quality has not been measured. The adapter follows the official [CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).

## Metrics and timing

`metrics.py` provides Recall@K, per-claim RR/RR@K, binary nDCG@K and macro reports. Recall divides by all relevant gold documents, including unretrieved papers. MRR is mean RR; untruncated MRR covers the supplied ranking only. nDCG uses binary gain and `1/log2(rank+1)` discount with ideal length `min(K, gold_count)`.

No-gold claims return null and are excluded with a count. Gold-labeled claims with empty/missing rankings count as zero. Unknown claim IDs and duplicate ranking IDs are rejected. No eligible claims produces null means.

Timing uses a monotonic CPU clock. Unknown upstream durations stay null. Internal ranking time excludes model loading, upstream retrieval and JSON output. Do not sum concurrent branches into end-to-end latency. Async/GPU synchronization and stable throughput measurement require separate target-host validation.

## SciFact and Person 2 export

```sh
.venv/bin/python -m ranking_evaluation.scripts.setup_scifact
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.scifact --corpus data/scifact/corpus.jsonl --claims data/scifact/claims_dev.jsonl --split dev --output outputs/ranking_evaluation/dev-template.json
```

The explicit download writes selected official files and a hash manifest under ignored `data/scifact/`. The official latest URL can change; actual hashes identify the version. The adapter follows the [official data schema](https://github.com/allenai/scifact/blob/master/doc/data.md), accepts labeled train/dev, validates sentence references and preserves numeric IDs as decimal strings.

Person 2 fills each template record's `bm25_ranking`/`sbert_ranking`, optional durations and top-level `retrieval_config`. Record actual method/tokenization/parameters, model revision, text/truncation policy, branch depth, device, batching and timing scope. Both branches must use the same corpus/original claims without gold-based candidate filtering. Preserve dataset hashes and every claim, then set `status="complete"` in a separate export file. Pending templates cannot be evaluated.

Proposed binary relevance includes documents with nonempty SUPPORT or CONTRADICT evidence; cited IDs alone are not positives. No-evidence claims remain present and are excluded from relevance macro means with counts. Team protocol agreement is pending. Live DOI/provider identities and SciFact IDs are separate namespaces.

```sh
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.scifact --corpus data/scifact/corpus.jsonl --claims data/scifact/claims_dev.jsonl --split dev --rankings completed-rankings.json --output outputs/ranking_evaluation/dev-bundle.json
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.evaluate --input outputs/ranking_evaluation/dev-bundle.json --output outputs/ranking_evaluation/dev-report.json
```

The bundle retains full corpus/gold IDs while materializing only ranked candidates. The runner compares supplied BM25/SBERT rankings with RRF and optional reranking; it does not implement or remeasure retrieval. Reports preserve per-claim output, metrics, branch depths, source/input hashes, Git state, environment and model-file hashes. Caller-declared model ID/revision is not a substitute for actual file hashes. Choose new output paths; overwrite is refused.

## Experiment grids and examples

```sh
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.experiment_suite --input outputs/ranking_evaluation/dev-bundle.json --config ranking_evaluation/experiments/rrf-pilot.json --output-dir outputs/ranking_evaluation/new-pilot
```

Pilot configs cover RRF constants 20/60/100 and optional depths 10/20/50; these are not optimized choices. Reranking requires explicit `--model-path`, `--model-id`, `--model-revision`; `--cpu-float64`, `--batch-size` and `--max-length` configure the CPU model.

The suite retains an input snapshot, config, manifest, reports/logs and CSV/JSON summaries. Baselines are listed once per repeat. Each rerank job loads its own model; stage timings retain cold inference. Replaying exported ranks does not create new retrieval timing observations. Summary p95 uses nearest rank across claim durations. One-repeat pilots are not stable latency evidence. Failed jobs preserve a failed manifest; existing output directories are refused. No automatic best-parameter selection occurs. Declare tuning/final evaluation splits before real experiments.

Offline synthetic entry points:

```sh
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.examples.fuse_mock
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.examples.evaluate_mock
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.examples.write_mock_benchmark outputs/ranking_evaluation/new-mock-bundle.json
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.examples.handoff_mock outputs/ranking_evaluation/new-handoff.json
```

Real CPU smoke check after setup:

```sh
PYTHONPATH=live_search/src HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m ranking_evaluation.examples.verify_cpu_reranker
```

Examples use fictional claims/rankings or constructed passages, not scientific experiment results. Model weights, datasets, local notes, environment snapshots and runtime reports are excluded from the module's publication set. All authored code and documentation use English.
