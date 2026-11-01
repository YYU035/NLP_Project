# Research scripts

Run these from the module directory after installation. Normal retrieval uses
`paper_search.retrieval`; these scripts prepare or analyse experiments.

| Scripts | Purpose |
|---|---|
| `query_examples.py`, `query_audit.py`, `query_audit_cases.py` | Generate query examples and compare reformulations |
| `provider_diagnostics.py` | Investigate provider syntax, metadata and access errors |
| `run_query_evaluation.py`, `compare_arxiv_concepts.py` | Compare original claims with concept queries |
| `experiment_openaire.py` | Compare original and spaCy queries on OpenAIRE |
| `experiment_keybert.py`, `experiment_arxiv_keybert.py` | Compare spaCy and KeyBERT retrieval |
| `evaluate_relevance.py` | Prepare and optionally run the separate Gemini relevance evaluation |
| `prepare_integration_readiness.py` | Prepare review samples and fixed experiment plans |
| `run_readiness_experiment.py` | Run the planned depth or arXiv relaxation comparison |
| `analyze_readiness_runs.py` | Join saved results with labels and export assessment files |

For example, reanalyse the saved depth and relaxation runs without API calls:

```powershell
.venv\Scripts\python.exe scripts/analyze_readiness_runs.py --depth artifacts/depth-approved-run --fallback artifacts/fallback-approved-run --output outputs/readiness-review
```

Use a new output directory. Completed results are in the
[experiment summary](../artifacts/README.md); no rerun is needed for integration.
Live retrieval commands require an explicit opt-in flag. The Gemini evaluator
has its own free-tier checks; it is not part of the retrieval pipeline.

For model setup, use `python -m paper_search.model_setup` rather than an
experiment script.
