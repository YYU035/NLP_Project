# Submission-readiness review — 2 November 2026

The finalized architecture remains **OpenAlex → KeyBERT; OpenAIRE → KeyBERT;
arXiv → spaCy; top 30 per provider; cross-provider deduplication; arXiv
fallback off by default**. No live APIs, model downloads, new evaluation runs,
Git commits or pushes were performed during this cleanup.

## Targeted changes

| Files | Change |
|---|---|
| `README.md` | Replaced outdated production description with finalized setup, configuration, handoff, tests and research navigation |
| `.env.example` | Final-policy flags (`OpenAIRE=true`, `Semantic Scholar=false`); clearly separated legacy limits/configuration |
| `.gitignore` | Removed blanket `artifacts/` exclusion; retained evidence while ignoring weights, credentials, caches, runtime databases and outputs |
| `pyproject.toml` | Corrected package description; dependencies and Python floor unchanged |
| `src/paper_search/keybert_queries.py`, `retrieval.py` | Anchored the existing default model location to the editable checkout, fixing calls from other working directories |
| `src/paper_search/model_setup.py` | Relocated the existing pinned-model download helper into standalone setup; no new model/strategy |
| `scripts/experiment_keybert.py` | Removed the old inline download implementation and its unused `os` import; old command delegates to the same helper |
| `examples/person2_integration.py` | Minimal package-only handoff with complete-outage handling |
| `tests/test_submission_readiness.py` | Five focused offline checks: model-path portability, mocked setup, actual public wrapper/provider parsers, example and Git ignore semantics |
| `docs/retrieval_integration.md` | Added standalone setup, explicit model-path injection and example instructions |
| `scripts/README.md`, `artifacts/README.md` | Separated reproducibility tools/evidence from the production entry point |
| This review | Records decisions, validation and remaining pre-push work |

**No source, experiment, label, report or response archive was deleted or
renamed.** The only removed function body was relocated rather than discarded;
`experiment_keybert.download_model` remains importable through its import.
Regenerable Python bytecode, pytest/Ruff caches and the temporary wheel-build
tree were cleaned: **11 generated directories / 157 files** removed after
validation, with every target verified inside the workspace. Virtual
environments, local model weights and editable
installation metadata remain locally available and ignored.

Ruff found no unused imports in the final tree. No debugger statements,
personal absolute paths or confirmed disposable experimental implementations
were found in the reviewable source/evidence. Historical scripts were checked
against tests, reports and their reproducibility purposes, not deemed unused
merely because they are absent from the final live path.

## Integration and behavior verification

Person 2 imports `paper_search.retrieval.retrieve` or `RetrievalService` after
editable installation, without importing `scripts`. A fresh isolated Python
process successfully imported the public API without experimental modules.
The minimal example is [examples/person2_integration.py](../examples/person2_integration.py).

The public wrapper was exercised offline with the real OpenAlex/OpenAIRE/arXiv
parsers and mocked transport/NLP outputs. It made exactly three requests, used
the agreed provider strategies and top-30 parameters, made no fallback calls,
merged a common DOI into one paper and retained all three provider origins.
The existing suite covers normalization, DOI/arXiv/title deduplication,
metadata, dates, partial/complete outages, retries, timeouts and relaxation.
No output-schema, scientific-query, deduplication or transport logic changed.

The portability fix changes where weights are looked up when a caller changes
directories; from the checkout root the resolved model and revision are the
same. Explicit generator injection remains available. Standalone wheels can
also use an explicitly configured absolute model path; the documented default
team workflow is editable installation from the checkout.

## Checks completed

- `python -m pytest tests integration_tests -q`: **268 passed, 6 skipped**.
  Run on **Python 3.13.5**; the declared Python 3.11 floor was not separately
  exercised here. All six skips are explicit live tests; no `--run-live` was used.
- `python -m ruff check .`: **passed**.
- `python -m pip check`: **no broken requirements**.
- Offline wheel build with `pip wheel . --no-deps --no-build-isolation
  --no-index`: **passed**. The wheel contains 43 files, includes the setup
  module and bundles no experiment scripts, artifacts, tests, weights or
  runtime databases.
- Public import in isolated Python: **passed**, no experimental-script imports.
- `compileall` syntax/import-compilation check: **passed**. Local Markdown-link
  audit found **no missing targets**. All hashes in the completed live-results
  evidence manifest still match; historical inputs/labels/responses were not
  rewritten by cleanup.
- Real Git ignore-pattern test in an isolated temporary repository: **passed**;
  research reports, labels and response archives remain eligible while `.env`,
  weights, databases and ad hoc outputs are excluded. The actual workspace was
  not initialized as a Git repository.
- Heuristic secret/path scan of 522 reviewable text files (about 67 MB before
  cleanup): no Google/GitHub token signatures, private-key blocks, long literal
  Bearer tokens or personal Windows user paths detected. This is not a full
  Git-history or secret-scanner audit.

## Retained compatibility and unresolved decisions

- Legacy `SearchPipeline`, Semantic Scholar, deterministic query ablations and
  their configuration remain because previous tests and experiments use them.
  They are not the finalized handoff entry point. Removing them would break
  compatibility or reproducibility rather than clean unused production code.
- `pdf_verification.py` is a documented empty extension point. Its future role
  is uncertain, so it was flagged and retained; PyMuPDF remains in the declared
  dependencies as originally requested. No PDF feature was added.
- The optional Gemini evaluator and its local cache are separate from
  retrieval. Saved prompts, datasets and manual judgments are retained;
  runtime SQLite files are ignored, not deleted.
- Model weights and historical environment/cache files remain local. The model
  downloader and pinned revision provide setup reproducibility; no weights
  belong in the source submission.
- Existing provisional relevance conflicts and unassessed deeper/fallback
  candidates are research limitations, not resolved by code cleanup. Keep
  fallback off; see the [completed experiment report](../artifacts/readiness-live-results-20261101/report.md).

## Before pushing in the actual Git clone

**This workspace has no `.git` directory.** It is not possible here to inspect
tracked files, staged changes, prior commits or accidentally committed secrets,
or to remove anything from an existing index. Updating `.gitignore` does not
untrack files already committed elsewhere.

In the actual clone, review `git status --short`, `git diff --check`, the staged
diff and `git ls-files` before committing. Verify that evidence under
`artifacts/` is included and `.env`, local weights, `.venv`, caches and runtime
databases are absent. If private files are already tracked, remove only those
verified paths from the index while preserving local files; assess any exposed
credential separately. Do not force-add the ignored model/cache directories.

No unnecessary source deletions or renames are proposed. The repository is
ready for team use through the documented editable install; the actual Git
index/history review is the remaining submission check.
