# Person 3 — Hybrid Ranking & Evaluation

Work branch: `yuyue`. See [development plan and work log](DEVELOPMENT_PLAN.md).

## Current scope

P1 defines a **proposed**, not yet team-approved Person 2/3 interface. It directly
reuses Person 1's `RetrievalResult` and `RetrievedPaper`; no copied paper schema
or network call is needed. RRF is implemented; evaluation and Cross-Encoder
are not implemented yet.

`RankingInput` contains a claim ID, the complete Person 1 result, two ordered
lists of preferred `identity` values, and optional measured branch durations.
The first list element is rank 1. Aliases must be resolved to preferred IDs
before passing rankings. Duplicate or unknown IDs and inconsistent claims are
rejected. Lists may cover different subsets and may be empty. An empty list
alone does not describe a failure; upstream diagnostics must be retained.
Unknown durations remain `None`, rather than a fabricated zero.

`RankedPaper` defines a future output entry with original metadata/provenance,
final rank, fusion score, branch ranks and optional Cross-Encoder score.

## Offline development

From the repository root, using Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'pydantic>=2.7,<3' 'pytest>=8,<10'
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.examples.mock_input
PYTHONPATH=live_search/src .venv/bin/python -m ranking_evaluation.examples.fuse_mock
PYTHONPATH=live_search/src .venv/bin/python -m pytest ranking_evaluation/tests -q
```

The lightweight schema/example workflow needs only Pydantic; pytest is for
validation. It reads Person 1's source via `PYTHONPATH` and does not install or
run its search/NLP dependencies. Live search itself requires Person 1's full
[setup](../live_search/README.md).

The example has fictional papers and manual rankings, no measured timings,
model output or relevance labels. It validates the interface only. No GPU,
model downloads or API keys are needed.

## RRF

```python
from ranking_evaluation.rrf import reciprocal_rank_fusion

# inputs is a validated RankingInput from the preceding interface.
ranked_papers = reciprocal_rank_fusion(inputs, rrf_k=60, top_k=10)
```

Each branch contributes `1 / (rrf_k + rank)`, counting from 1. A missing
branch contributes zero. Only papers appearing in at least one ranking are
returned. Exact score ties use ascending lexical paper identity for reproducible
output. Empty branches produce an empty result; upstream diagnostics still
determine whether retrieval failed.

`rrf_k` must be a non-negative integer (default 60); it is distinct from the
output limit `top_k`, which accepts a non-negative integer or `None` for all
results. Zero output limit returns no entries. Input branch rankings are not
truncated before fusion. Output entries preserve full Person 1 records and both
branch positions. No calibrated probability or scientific verdict is produced.

The synthetic example yields A, B, C, with scores respectively
`1/61 + 1/62`, `1/63 + 1/61`, and `1/62 + 1/63`. This demonstrates arithmetic
only; it is not evidence that fusion improves real retrieval quality.
