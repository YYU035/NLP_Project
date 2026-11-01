# Preserved research evidence

These files support academic grading, label reconciliation and reproducibility.
They describe historical runs; their recommendations may predate the finalized
OpenAlex/KeyBERT, OpenAIRE/KeyBERT, arXiv/spaCy policy. They are not separate
experiments merely because reports or derived analyses repeat their records.

| Artifact | Evidence |
|---|---|
| `query-evaluation-20261030/` | Original/concept search coverage, controls and responses |
| `provider-diagnostics-20261030/`, `provider-diagnostics-followup-20261030/` | Provider access/syntax diagnostics and archived responses |
| `arxiv-concepts-20261031/` | Full-claim/spaCy comparison; original 250-row provisional labels |
| `gemini-relevance-20261031/` | Blinded dataset, prompt, manual check; API judgments were not run |
| `openaire-experiment-20261101/` | Original/spaCy OpenAIRE results and unresolved assessment exports |
| `keybert-experiment-20261101/` | spaCy/KeyBERT OpenAlex/OpenAIRE comparison, labels and final analysis |
| `arxiv-keybert-20261101/` | arXiv spaCy/KeyBERT comparison and assessment export |
| `integration-readiness-20261101/` | Frozen plans, 12 conflicts and 47-example review sample |
| `depth-approved-run/` | Actual frozen depth run: ten top-30 requests/provider, 30 total |
| `fallback-approved-run/` | Actual frozen arXiv run: ten primary + five fallback requests |
| `readiness-live-results-20261101/` | Offline analysis of the preceding two runs; assessment exports |
| Root `*.json` | Earlier smoke tests and local query audits |

Keep CSV labels, manifests, plan versions, raw response archives, checksums,
query drafts, reports and mappings together. Unrun drafts are evidence of
pre-retrieval development, not additional live experiments. Numeric canonical
identities depend on the frozen historical record order; never join labels by
row number, title alone or rank. Unknown/conflicting labels are not irrelevant.

`local-models/` is ignored: download the pinned weights using
`python -m paper_search.model_setup`. Runtime SQLite caches and `runtime/` are
also ignored; saved CSV/JSON judgments and reports remain eligible for Git.
The local Gemini cache currently contains no API judgments; it is not a label
source needed for the recorded evaluation. New ad hoc outputs should go under
the ignored repository-level `outputs/`, and be reviewed before being retained
as formal research evidence.

Start with the [latest results](readiness-live-results-20261101/report.md).
No further live calls are required to review the saved candidate papers.
