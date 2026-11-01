# Research commands (not the production entry point)

Person 2 should import `paper_search.retrieval`, not these scripts. Run scripts
from the repository root after editable installation. Preserve their paths:
tests, reports and frozen experiment instructions refer to them.

| Script | Purpose |
|---|---|
| `query_examples.py` | Original deterministic-query examples, no retrieval |
| `query_audit.py`, `query_audit_cases.py` | Recorded before/after query audit and claims |
| `run_query_evaluation.py` | Original/concept coverage experiment and controls |
| `provider_diagnostics.py` | Historical provider syntax/access diagnosis |
| `compare_arxiv_concepts.py` | Legacy full-claim vs arXiv concept rendering |
| `experiment_openaire.py` | Original vs spaCy OpenAIRE experiment |
| `experiment_keybert.py` | Matched spaCy/KeyBERT experiment; old model-setup command retained |
| `experiment_arxiv_keybert.py` | Matched arXiv spaCy/KeyBERT experiment |
| `evaluate_relevance.py` | Separate free-only Gemini evaluation tooling |
| `prepare_integration_readiness.py` | Offline conflict audit, blinded sample and frozen plans |
| `run_readiness_experiment.py` | Explicit opt-in depth/fallback experiments |
| `analyze_readiness_runs.py` | Offline label joins, counts and assessment exports |

These are reproducibility tools, not automatically run on import. Some commands
can overwrite generated reports; follow the corresponding protocol and use a
fresh output directory. Live commands require explicit opt-in. Completed
experiments do not need to be repeated for setup or integration.

Model setup is now independent: `python -m paper_search.model_setup` downloads
the same pinned weights without preparing or running an experiment. The old
`experiment_keybert.py download-model` command delegates to that helper.

See the [artifact index](../artifacts/README.md), the
[integration contract](../docs/retrieval_integration.md) and the root README.
