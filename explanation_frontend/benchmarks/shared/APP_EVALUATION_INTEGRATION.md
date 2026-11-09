# App Handoff: Three Fixed Evaluation Datasets

The selected set is **SciFact (biomedical), CSFCube (computer science), RECIPER
(materials/EEE-related research)**. Read `shared/datasets/evaluation_catalog.json`
for the authoritative list of display names, task/split labels, result files,
metric columns and limitations. All paths are relative to the repository root.

## Recommended integration

1. Fetch the `yuyue` branch and integrate the reviewed changes into the app branch.
   The branch already includes the previously merged contributor source modules.
2. Use the committed CSV/JSON result summaries for an evaluation/results view.
   This does not require downloading raw corpora or initializing neural models.
3. Display each dataset's domain, task, split, query unit, query count and metric
   denominator. Keep dataset results separate; do not compute an overall accuracy
   from retrieval, grading and sentence-evidence scores.
4. Normal scientific-claim interactions continue through live search, retrieval,
   evidence and verification. Do not route every claim through the benchmark
   collections just because the evaluation view lists three datasets.
5. If adding a rerun feature, run the documented CLI jobs separately and retain
   their status/output directories. Benchmark setup and long model runs should
   not silently run when rendering an app page.

## Reading the committed results

Python's standard library is sufficient for a lightweight result reader:

```python
import csv
import json
from pathlib import Path

# Set this to the cloned repository root in the app's own configuration.
repo_root = Path("/path/to/NLP_Project")
catalog = json.loads(
    (repo_root / "shared/datasets/evaluation_catalog.json").read_text()
)
for dataset in catalog["datasets"]:
    with (repo_root / dataset["ranking_summary_path"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Keep strings or convert only the numeric metric columns the view uses.
    # Display dataset["task"], dataset["split"] and dataset["limitations"] too.
```

Use the catalog's `method_key_columns`: SciFact has multiple Cross-Encoder depths,
so `method` alone is not a unique row key. CSFCube distinguishes binary nDCG from
`graded_nDCG`. RECIPER measures recovery of its single known source paper, and
all 1,024 questions face the same full 343-paper collection.

SciFact's `evidence_summary_path` points to the existing sentence-selection
summary. Preserve oracle-paper versus retrieved-paper conditions and the report's
precision/recall/F1 definitions. Final claim-verification Accuracy is a different
quantity; it must not be inferred from paper Recall or evidence F1.

## Backend and data boundary

Adapters: `shared/datasets/csfcube.py`, `shared/datasets/reciper.py`; runner/scorer
entry points: `ranking_evaluation.run_csfcube`, `ranking_evaluation.run_reciper`
and their `evaluate_*` modules. Both invoke the actual P2 BM25/SBERT implementations
and P3 `RankingInput` / `rank_claim` boundary. Model-facing text and evaluator-only
gold are exported separately.

The existing frontend has pinned vendored runtime modules. Merging root source
files does not automatically refresh those copies. If runtime model behavior is
also updated, review that synchronization separately using the frontend's own
integration instructions; displaying the new benchmark summaries does not require
changing the live-search runtime or importing benchmark gold into it.

See [dataset setup](datasets/README.md), [CSFCube report](datasets/CSFCUBE_REPORT.md),
[RECIPER report](datasets/RECIPER_REPORT.md) and
[Person 3 handoff](EVALUATION_HANDOFF.md).
