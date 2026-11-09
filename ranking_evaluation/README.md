# Hybrid Ranking and Evaluation

Person 3 implementation: RRF fusion, optional Cross-Encoder reranking, source-linked evidence selection, retrieval/evidence metrics and reproducible experiments. Person 2's pinned BM25/SBERT implementation has been integrated and evaluated on the complete SciFact dev split. Local CPU/GPU smoke checks and frontend integration checks are available. Scientific evaluation results are development evidence, not untouched final-test claims.

## Additional fixed evaluation datasets

The active domain evaluation set is SciFact, CSFCube and RECIPER. Dataset
adapters/download manifests are in `shared/datasets/`; runners and scorers are
in this module. Fixed evaluation collections do not constrain normal live search.

- CSFCube: `run_csfcube.py` / `evaluate_csfcube.py`, expert graded paper-pool ranking.
- RECIPER: `run_reciper.py` / `evaluate_reciper.py`, all 1,024 questions against
  the complete 343-paper collection, using the team's title/abstract representation.
- [RECIPER protocol, commands and results](../shared/datasets/RECIPER_REPORT.md).
- [Current cross-dataset status](../shared/datasets/EVALUATION_STATUS.md).

RECIPER source targets are generated-QA provenance labels, not expert sentence
evidence or exhaustive relevance judgments. Keep its source-paper recovery metrics
separate from SciFact evidence F1. Matter-of-Fact and ClimateCheck scripts/results
are archived exploration, outside the selected three-dataset result table.

The Cross-Encoder adapter supports explicit CPU, CUDA and Apple MPS devices;
CUDA/MPS timing includes device synchronization. CPU float64 remains an opt-in
workaround for the previously observed local CPU nonfinite-score issue. The
actual run report records device, dtype, model files and software versions.

## Setup and tests

Run from the repository root with Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'pydantic>=2.7,<3' 'pytest>=8,<10'
PYTHONPATH=live_search/src .venv/bin/python -m pytest ranking_evaluation/tests -q
```

The lightweight path only imports Person 1's schemas from `live_search/src`; live search itself requires its own setup. Optional NumPy/scikit-learn reference checks skip when those packages are absent. Python 3.13.1 was tested locally. Python 3.11 and a real float32 A40 smoke check also passed on the NTU EEE cluster; this does not establish benchmark quality.

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

Timing uses a monotonic wall clock. Unknown upstream durations stay null. Internal ranking time excludes model loading, upstream retrieval and JSON output. Do not sum concurrent branches into end-to-end latency. Async/GPU synchronization and stable throughput measurement require separate target-host validation.

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

Pilot configs cover RRF constants 20/60/100 and optional depths 10/20/50; these are not optimized choices. Reranking requires explicit `--model-path`, `--model-id`, `--model-revision`; `--device cpu|cuda`, `--batch-size` and `--max-length` configure inference. `--cpu-float64` is restricted to CPU. CUDA scorer calls synchronize before and after inference.

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

## Cluster GPU smoke check

`examples.verify_gpu_reranker` checks CUDA availability, pinned model file hashes,
finite scores, and the expected first passage on three constructed pairs. It
records cold and warm inference separately with CUDA synchronization. This is
an environment check, not SciFact evaluation or stable latency evidence.

On the NTU EEE cluster, provision an independent environment and pinned model
under your SSD project storage, following the official
[Slurm guide](https://github.com/NTUEEECluster/docs/blob/main/slurm-guide.md) and
[storage guide](https://github.com/NTUEEECluster/docs/blob/main/storage-guide.md).
The batch script expects `$RANKING_RUNTIME/env/bin/python` and the model plus
`download_manifest.json` under `$RANKING_RUNTIME/model`. Redirect package,
Hugging Face, and temporary caches to SSD before setup. Then, from the repository
root:

```sh
export RANKING_RUNTIME=/projects/YOUR_SSD_PROJECT/nlp-ranking
export RANKING_REPO="$PWD"
mkdir -p outputs/ranking_evaluation
sbatch ranking_evaluation/scripts/gpu_smoke.sbatch
squeue --me
```

The script requests one A40 for at most ten minutes. Pending jobs wait for
resources; they have not run inference. Logs and reports use the Slurm job ID
under `outputs/ranking_evaluation/`. The benchmark CLI defaults to CPU; pass `--device cuda` with model options to
use CUDA. The independent smoke check explicitly uses CUDA.


## Person 2 integration and full-corpus GPU comparison

`export_naila` runs unchanged BM25/SBERT scoring implementations from a pinned
local snapshot of Naila's retrieval module. The snapshot root must include
`retrieval/` plus `source_manifest.json` with `commit` and `files_sha256` fields;
the latter maps relative source filenames to SHA-256 hashes. All listed source
files are checked before import. Provision a local SentenceTransformer model
explicitly, including `download_manifest.json` with `model_id`, `revision` and
`files_sha256`. Declared model identity and all manifest hashes are checked.
Neither inference path downloads implicitly.

```sh
PYTHONPATH=live_search/src python -m ranking_evaluation.export_naila \
  --retrieval-root /path/to/pinned-naila-snapshot \
  --corpus /path/to/scifact/corpus.jsonl \
  --claims /path/to/scifact/claims_dev.jsonl --split dev \
  --model-path /path/to/local-sbert \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --model-revision YOUR_PINNED_REVISION --device cuda \
  --k 100 --output outputs/ranking_evaluation/new-rankings.json
```

Unlike upstream's default benchmark loader, this adapter retains every claim,
including no-evidence claims. It scores the complete corpus for each original
claim and exports both branches independently. The existing `scifact` CLI then
converts this export to a benchmark bundle. No gold-based candidate filtering
is performed. Real NLTK Porter stemming is required. Index/model loading and
index construction are recorded separately from warmed query scoring.

The integration sorts all scores before taking top K: descending score, then
ascending corpus position. This implements upstream's documented tie rule and
avoids its known `argpartition` cutoff-tie gap. It is an explicitly recorded
integration change; upstream source remains unchanged. SciFact sentence lists
are retained by Person 2's loader; the paper-ranking bundle joins abstract
sentences for document scoring and must not be used as a sentence-index export.

`scifact_gpu.sbatch` runs an offline CUDA smoke check, Person 2 export, bundle
conversion and the declared RRF/rerank pilot on one GPU. Set `RANKING_RUNTIME`
(SSD runtime root), `RANKING_REPO` and `NAILA_SOURCE`, then submit from the repo
root after creating `outputs/ranking_evaluation`. Runtime expects `env/`,
`model/` (Cross-Encoder), `sbert-model/` and `scifact/` under the SSD root.
The script freezes Python sources, a protocol and package versions before
inference. Large outputs stay on SSD with a link under repository outputs.

This comparison is **development evaluation**, not untouched final testing:
SciFact dev has already been used by teammates. The fixed pilot compares
RRF k=60 and rerank depths 10/20/50 without selecting a best configuration.
One repeat includes cold inference and cannot establish stable GPU latency.

## Evidence sentence selection and evaluation

`evidence_selection.select_evidence` provides deterministic sentence and adjacent
sentence-window BM25 baselines over the **original sentence list**. It never
re-splits SciFact abstracts or reads gold labels. Each span retains paper ID,
original sentence indices, exact joined-abstract character offsets and a
text-derived passage ID. These baselines use casefolded Unicode word tokens,
no stemming/stopword removal, and fixed k1=0.9/b=0.4. They differ from the
frontend's provisional regex selector and do not replace it automatically.

`evaluate_evidence` compares three fixed policies: three single sentences,
six single sentences, and three windows of at most two sentences. The latter
two have the same maximum sentence budget, although overlapping windows can
cover fewer unique sentences. Oracle-gold-paper evaluation isolates selection;
retrieved-paper evaluation uses every top-K paper, without filtering by gold.
It reuses frozen ranking reports rather than rerunning retrieval or a model.

```sh
PYTHONPATH=live_search/src python -m ranking_evaluation.evaluate_evidence \
  --corpus /path/to/scifact/corpus.jsonl \
  --claims /path/to/scifact/claims_dev.jsonl --split dev \
  --ranking-reports /path/to/ranking-report.json \
  --paper-depths 5 10 --output-dir outputs/ranking_evaluation/new-evidence-run
```

The optional `--frontend-root /path/to/kireeti-checkout --policies frontend_current`
evaluates the existing frontend selector without changing it. Run it in the
frontend's own environment: its Porter/simple-stemmer fallback depends on
whether NLTK is installed. The adapter validates source identity and exact text,
merges selected character intervals and maps them back to original SciFact
sentences. Partially covered original sentences receive no complete-sentence
credit and are retained in per-claim diagnostics.

Reports include micro/macro sentence Precision/Recall/F1 against the union of
annotated rationale sentences, per-document recovery of **any complete valid
rationale**, and claim-level complete-rationale recovery. Multiple alternative
rationales are preserved, never selected using gold for inference. Overlapping
windows are deduplicated for sentence scoring. Unretrieved gold sentences are
misses; selected sentences in non-gold papers remain in precision denominators.
No-gold claims remain in output and are excluded from relevance means with counts.
Their selected-sentence counts are reported without inventing independent error
labels. Dataset hashes and exact claim coverage must match the ranking reports.

These are **custom sentence-selection development diagnostics**, not official
SciFact evidence scoring, NLI accuracy, scientific verification, or untouched
final testing. Gold unions can penalize an otherwise sufficient alternative
rationale; the separate complete-rationale metric makes that difference visible.
Context neighbours and genuinely relevant but unannotated sentences can count
as false positives. No PDF extraction or downstream token-truncation evaluation
is implied. Every run preserves its protocol, source/input hashes, summary,
readable report and source-linked per-claim passages in a new output directory.

`ranking_evaluation/scripts/evidence_eval.sbatch` runs the three native baselines
on CPU using the existing ranking-comparison outputs. It requires
`RANKING_RUNTIME`, `RANKING_REPO` and `RANKING_COMPARISON`, and stores the run on
SSD with a link under repository outputs. No GPU or new model download is needed.


## Semantic evidence selection and Person 4 handoff

`semantic_evidence` ranks original sentences with a local normalized MiniLM
encoder (maximum 256 input tokens for scoring). A frozen cosine threshold can
return no passages. Optional context includes one original sentence on each
side of a selected anchor; overlapping windows merge. Original text remains
whole and source-linked. Person 4 remains responsible for reporting/excluding
NLI inputs that exceed its token limit. Cosine scores are relevance values.

`semantic_experiment` searches top-k, cosine threshold and context settings on
200 deterministically SHA-256-sampled **train** claims with real full-corpus
BM25/SBERT/RRF top-5 retrieval. It freezes the selected configuration before
scoring dev. Its sentence/context ablations retain the same threshold and
anchor budget. No pretrained weights are updated, no dev labels select settings,
and no final-test claim is made: the team already used dev for development.

Run `scripts/semantic_evidence.sbatch` with `RANKING_RUNTIME`, `RANKING_REPO`
and `NAILA_SOURCE`. It preserves the full source snapshot, train grid/results,
train rankings, frozen configuration, model/data hashes, per-claim passages and
dev summaries on SSD. This is a CPU job; embeddings are cached by model/data/
package fingerprint and contain no gold labels.

For a completed RankingOutput, produce the native Person 4 input dictionaries:

```sh
PYTHONPATH=live_search/src python -m ranking_evaluation.select_evidence \
  --input ranking-output.json --config /path/to/frozen-config.json \
  --model-path /path/to/local-sbert --verifier-tokenizer /path/to/local-nli --paper-depth 5 \
  --output outputs/ranking_evaluation/new-evidence-handoff.json
```

`person4_passages` emits the exact `passage_id`, `paper_id`, `section`, `text`,
`score` fields of Person 4's EvidencePassage. The enclosing handoff retains the
original claim, upstream paper metadata, source spans and per-paper selected/
no-passage status. No empty selection is converted into a scientific verdict.
SciFact callers supply original sentence lists directly. Live abstract callers
use explicitly heuristic sentence boundaries and exact original text offsets.

### Optional local frontend integration

```sh
python -m ranking_evaluation.scripts.install_frontend_adapter \
  --frontend-root /path/to/kireeti-checkout
export PERSON3_EVIDENCE_CONFIG=/path/to/frozen-config.json
export PERSON3_EVIDENCE_MODEL_PATH=/path/to/local-sbert
export PERSON3_VERIFIER_TOKENIZER_PATH=/path/to/local-nli
```

Then launch the frontend in its own environment. The installer copies the
four evidence runtime modules under `explanation_frontend/_person3`, records
source hashes, and adds an opt-in hook at the existing selector boundary.
The bridge checks model identity and actual file hashes against the frozen
configuration before offline loading, serializes inference across sessions,
and emits native Person 4 objects. Run diagnostics identify the enabled stage.
Without the config environment variable, the existing selector is retained.
No upload can choose a Python factory, model directory or training settings.
The installer does not push commits or alter teammate remote branches.


## Completed Person 3 delivery

See [PERSON3_HANDOFF.md](PERSON3_HANDOFF.md) for the implementation checklist,
measured development comparisons, frontend activation and exact remaining
team-owned evaluation boundaries. `experiments/semantic-train-frozen.json`
contains the train-selected deployment settings. The full training search and
dev artifacts are preserved outside source control with model/data hashes.
`rrf_ablation` reproduces constants 20/60/100 using compact ID rankings without
repeating large paper metadata. Semantic benchmark selection times use cached
embeddings/scores and must not be presented as online neural-inference latency.

## PDF/full-text completion checklist and actual verifier budget

The current Person 4 branch is `chong_yao_claim_verification`, inspected at
`1ac06b675e3d0719a63cb8efe98eda4f2c5fd576`. Its native verifier takes evidence as
premise and the verbatim claim as hypothesis, with **512 total pair tokens,
including special tokens**. The local deployment uses its pinned DeBERTa
tokenizer revision `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`.
`verifier_budget.enforce_verifier_budget` uses that actual tokenizer with
`truncation=False`. Whole over-budget passages are rejected with explicit
reasons; evidence text and claims are never shortened. An all-rejected frontend
request is an operational `VERIFIER_INPUT_LIMIT`, not a scientific verdict.
Run filters retain every check/rejection. Set `PERSON3_VERIFIER_TOKENIZER_PATH`
in addition to the two semantic model/config variables before launching the
new frontend adapter.

`select_evidence` now also requires `--verifier-tokenizer /path/to/local-nli`.
The complete frozen dev prediction audit checked 39,308 passage instances;
maximum accepted pair length was 484 tokens. None required rejection, so the
reported benchmark metrics were unchanged. Deliberate 513-token boundary and
907-token real-tokenizer stress cases verify explicit rejection. Actual Person
4 inference on the separate PDF sample reported zero truncated accepted inputs.

`pdf_evidence.extract_pdf` accepts a local PDF and retains one-based PDF page,
exact character offsets into each extracted page string, PDF/page hashes and
extractor version. `document_from_pages` accepts already extracted full text
from Person 1. Page text uses PDF content-stream order; positional sorting was
observed to interleave columns in BERT and is not used. Scanned/encrypted PDFs
fail explicitly. Tables, figures, cross-page context and reading order still
require the separate review sample. No OCR is claimed.

```sh
python -m ranking_evaluation.select_pdf \
  --pdf /path/to/paper.pdf --paper-id YOUR_PAPER_ID \
  --claim-file /path/to/original-claim.txt --source-url https://source.example/paper \
  --config ranking_evaluation/experiments/semantic-train-frozen.json \
  --model-path /path/to/local-sbert --verifier-tokenizer /path/to/local-nli \
  --context-radius 1 --output outputs/ranking_evaluation/new-pdf-handoff.json
```

PDF context expansion is explicit; at most three accepted passages are returned
per paper, matching Person 4's vote cap. Exact source text and page references
remain alongside native EvidencePassage records. `passage_reranker` separately
implements optional passage-level Cross-Encoder reranking; its on/off experiment
uses a common semantic candidate pool and identical top-1 per-paper output
budgets, checking both actual model tokenizers without truncation. Document-level
and passage-level reranking results must be labelled separately.

`failure_analysis` produces annotated cases for relevant papers with irrelevant
selected sentences, incomplete multi-sentence rationales and context-rescued
rationales. Under the frozen RRF top-5 protocol it observed 35, 3 and 23
claim-document cases respectively, plus 28 missing-gold-document cases. These
categories can overlap and do not establish independent scientific error labels.

`shared_results` produces method-owner/experiment-runner separated CSV/JSON
plus owner-submission rows. Teammates run/confirm their own experiments; blank
confirmations are not treated as approvals.

### Required manual PDF review

The separate packet contains two fixed-version PDFs, six constructed claims,
twelve selected passages, original page images and a blank review CSV. These
are not an additional annotated benchmark dataset. `index.html` lets a human
reviewer compare the originals and download completed yes/no/uncertain ratings.
AI visual prechecks are separately identified and never fill human labels.

After an actual reviewer supplies the completed CSV:

```sh
python -m ranking_evaluation.scripts.validate_pdf_review \
  --packet /path/to/pdf-review-packet \
  --review-csv /path/to/human-review-completed.csv \
  --output /path/to/new-review-record.json --record-in-packet
```

This validates sample identities, unchanged source text, named reviewer/time and
all required ratings. Negative/uncertain ratings are valid and remain visible.
**The task cannot be marked fully closed while human review remains pending.**


The separate review packet can be reproduced with
`scripts.prepare_pdf_sources --output-dir /path/to/new-pdf-sources`, then
`scripts.build_pdf_review` with the config, local models/tokenizer, Person 4
checkout and a new output directory. The builder runs Person 4's unchanged
classify/verify methods with an explicit offline initializer, emits page images
and exact quotes, and generates the blank review form. Pin extraction/library
versions when comparing PDF offsets across machines.

## Compact submitted results

See [results/person3-v2](results/person3-v2/README.md) for retrieval and evidence-selection summaries, the frontend adapter patch, and the explicitly AI-assisted PDF review draft. Human PDF sign-off and teammate confirmations remain pending. Additional NFCorpus/TREC-COVID evaluations have not been run.

For PDF extraction, install PyMuPDF in addition to the semantic-model dependencies: `python -m pip install PyMuPDF`. The recorded PDF experiment used the version reported in each extraction manifest; layout differences across versions require rechecking source offsets.
