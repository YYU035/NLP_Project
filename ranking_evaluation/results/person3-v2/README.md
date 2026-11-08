# Person 3 compact results

These are development-evaluation summaries, not untouched test results. Full model weights, datasets, raw ranking outputs, PDFs and caches are excluded from Git. Reproduction commands and dependencies are in ../../README.md. Absolute paths in provenance manifests identify the original cluster run; replace them with your local paths when rerunning.

- ranking-summary.csv: paper retrieval/reranking metrics on SciFact dev.
- semantic-dev-summary.json: evidence selection and context comparison.
- retrieval-results.csv: method ownership is separate from experiment runner.
- owner-submissions.csv: owner confirmations are pending.
- AI_PDF_REVIEW.md and ai-pdf-review.csv: AI-assisted review only, not human gold labels. Human PDF review remains pending.
- frontend-budget-adapter.patch: frontend integration changes for the documented pinned Kireeti source; install_frontend_adapter.py installs the runtime modules.

Additional NFCorpus/TREC-COVID evaluation has been discussed but not implemented or run. No results for those datasets are claimed.
