# Fixed Academic Evaluation Datasets

The active evaluation set is **SciFact, CSFCube and RECIPER**. These are fixed
benchmarks for different tasks and domains. Normal application claims continue
through Person 1's live scholarly APIs, independently of these collections.

| Dataset | Domain | Evaluation and gold | Measured scope |
|---|---|---|---|
| SciFact | Primarily biomedical | Claim-driven paper retrieval; sentence evidence precision/recall/F1 and rationale recovery | Existing complete dev experiment; ranking uses 188 positive-eligible claims and reports 112 exclusions |
| CSFCube | Computer science | Graded expert paper-pool ranking; Recall/MRR/binary and graded nDCG | 24 fold1-dev query/facet tasks |
| RECIPER | Materials science, including EEE-related research | Known source-paper retrieval with generated questions | All 1,024 questions against the complete 343-paper title/abstract collection |

Do not pool incompatible scores into an overall system accuracy. CSFCube and
RECIPER lack expert sentence-evidence and final-verdict labels. RECIPER has one
known source target, not exhaustive human relevance judgments; its source-paper
Recall@K is not a measure of every scientifically relevant paper. Final verdict
Accuracy/Macro-F1 requires actual verification inference on a compatible task.

## Files for app integration

[The evaluation catalog](evaluation_catalog.json) lists exactly these three
benchmarks, their task/domain labels, committed result paths, metric columns and
limitations. [The app integration handoff](../APP_EVALUATION_INTEGRATION.md)
explains how to consume the summaries without changing normal live search.
No raw dataset or model download is needed just to display committed results.

## Provision data explicitly

From the repository root:

```bash
.venv/bin/python -m pip install -r ranking_evaluation/requirements-datasets.txt
.venv/bin/python -m shared.datasets.setup csfcube
.venv/bin/python -m shared.datasets.setup reciper
.venv/bin/python -m shared.datasets.audit --output outputs/datasets/new-data-audit.json
```

The dependency file preserves the measured NumPy pin on Python 3.12+ and
selects a compatible NumPy range on Python 3.11. Exact local measurements used
Python 3.13.1; record changed package versions when rerunning elsewhere.
Displaying committed evaluation tables needs only Python standard-library readers.

Use the existing SciFact setup instructions in `ranking_evaluation/README.md`.
`ranking_evaluation.scripts.setup_scifact` records the downloaded file hashes;
its upstream `latest` URL is mutable, so compare those hashes with the recorded
experiment before claiming an exact data-version reproduction.

CSFCube and RECIPER setup pins official repository revisions and records SHA-256
checksums. Existing data is reused only after verification; unverified files are
not overwritten. Downloads are explicit, staged and limited to 50 MiB per file.
Raw files live under Git-ignored `data/`. Imports do not access the network.
CSFCube declares CC BY-NC 4.0; RECIPER's release declares MIT. Preserve original
attribution and rights for scientific source text.

## Prepare model-facing inputs separately from gold

```bash
.venv/bin/python -m shared.datasets.csfcube \
  --split fold1_dev --facet all --output-dir outputs/datasets/new-csfcube-inputs
.venv/bin/python -m shared.datasets.reciper \
  --output-dir outputs/datasets/new-reciper-inputs
```

Each creates a new directory and separates model-facing `retrieval-input.json`
from evaluator-only `judgments.json`. CSFCube has per-query judged pools and
numeric grades; RECIPER uses the full collection and a single source-paper target.
Their shared scoring boundary is `RankingInput` -> `rank_claim`, while dataset
adapters preserve the distinct benchmark protocols and label formats.

CSFCube query text includes its facet and original title/abstract; query-paper
self-matches are excluded. Binary positives use grade >=2; supplemental graded
nDCG uses linear gain/log2 discount. RECIPER uses unchanged questions and original
titles/abstracts only; generated answers, topics and recipes are excluded from
features. Both use actual Person 2 BM25/SBERT and Person 3 RRF/Cross-Encoder code.

## Run or inspect experiments

- [CSFCube commands, protocol and measured results](CSFCUBE_REPORT.md)
- [RECIPER commands, protocol, measured results and independent audit](RECIPER_REPORT.md)
- [Cross-dataset status](EVALUATION_STATUS.md)
- Existing SciFact reports: `ranking_evaluation/results/person3-v2/`

Neural runners require explicitly provisioned local weights. The pinned setup
helpers are `ranking_evaluation.scripts.setup_semantic_model` (configuration:
`ranking_evaluation/experiments/semantic-train-frozen.json`) and
`ranking_evaluation.scripts.setup_reranker`. Use a fresh output directory for each
run. Store dataset/model/code versions, device/dtype and metric denominators.

Compact CSV/JSON summaries are committed. Full query rankings, raw data and
weights remain outside Git. Single-run stage latency is not stable full-app
throughput. Tests verify contracts/arithmetic; they do not prove expert evidence
correctness or final scientific-verdict accuracy.

Optional Matter-of-Fact and ClimateCheck adapters/runners are retained as
exploratory compatibility code. They are not active benchmark entries in the
catalog and should not be presented as part of the selected three-dataset result
set. Personal selection notes and exploratory local reports are not needed by
the app integration.
