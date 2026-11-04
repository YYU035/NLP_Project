# Person 3 — Hybrid Ranking & Evaluation

Work branch: `yuyue`. See [development plan and work log](DEVELOPMENT_PLAN.md).

## Current scope

P1 defines a **proposed**, not yet team-approved Person 2/3 interface. It directly
reuses Person 1's `RetrievalResult` and `RetrievedPaper`; no copied paper schema
or network call is needed. RRF and evaluation are not implemented yet.

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
PYTHONPATH=live_search/src .venv/bin/python -m pytest ranking_evaluation/tests -q
```

The lightweight schema/example workflow needs only Pydantic; pytest is for
validation. It reads Person 1's source via `PYTHONPATH` and does not install or
run its search/NLP dependencies. Live search itself requires Person 1's full
[setup](../live_search/README.md).

The example has fictional papers and manual rankings, no measured timings,
model output or relevance labels. It validates the interface only. No GPU,
model downloads or API keys are needed.
