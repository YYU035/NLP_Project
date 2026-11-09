# Matter-of-Fact EEE: Claim-Only Development Baseline

> Archived exploratory experiment; this dataset is not currently selected as the third primary benchmark.

Date: 2026-11-09. Status: measured diagnostic baseline; system evaluation pending.

## Purpose and boundary

This experiment provides a reproducible third-dataset model baseline and tests
whether feasibility labels can be partly predicted from claim wording alone.
It does **not** evaluate the project's BM25/SBERT/RRF retrieval, evidence
selection, DeBERTa verification, or grounded explanations. Do not use this number
as evidence that the project retrieves relevant papers or correct evidence.
Three datasets now have measured models, but three system evaluations are not
complete. Instructor acceptance of different task-specific evaluations remains
an interpretation, not a confirmed grading decision.

## Frozen development protocol

- Dataset: pinned Matter-of-Fact release recorded by the existing adapter manifest.
- Scope: batteries, semiconductors, superconductors (EEE-related materials).
- Train: 936 claims from the released 2022 training split.
- Evaluate: all 1,800 selected claims from the released 2023 validation split.
- Features: claim text only, word TF-IDF unigrams/bigrams, minimum document
  frequency 2, at most 50,000 features, sublinear term frequency.
- Classifier: LogisticRegression, C=1, max_iter=2000, random_state=0.
- Decision: feasible probability strictly greater than 0.5. No parameter search.
- The vectorizer is fit only on training data. Gold IDs, source paper IDs, topic
  metadata, dates, supporting facts and explanations are excluded from features.
- Source paper overlap (ignoring arXiv versions): zero. Exact claim-text overlap:
  zero. Related wording and true/false variants can still affect difficulty.
- The 2024–2025 test split has not been scored by this runner.

## Actual validation results

| Metric | Value |
|---|---:|
| Accuracy | 0.642222 |
| Macro-F1 | 0.640678 |
| Evaluation claims | 1,800 |
| Balanced-class constant-prediction accuracy reference | 0.500000 |

All cases receive an actual classifier prediction, with no evidence citations.
The released labels define feasibility, not NLI entailment. This model cannot
explain feasibility from historical literature. The scorer deliberately reports
`formal_temporal_protocol_verified=false`; no claim of temporal-protocol
certification or original-paper baseline parity is made.

## Reproduce and inspect

Run from the repository root with scikit-learn available:

```bash
.venv/bin/python -m ranking_evaluation.run_matter_baseline \
  --split validation --scope eee \
  --output-dir outputs/datasets/matter-eee-claim-baseline-validation
```

Use a new output directory for another run. The runner rejects overwriting.
Artifacts in that directory: `predictions.json`, `evaluation.json`,
`protocol.json`, `timing.json`, `environment.json`, `provenance.json`.
These include per-case probabilities, model predictions, correctness, dataset
versions, code hashes, hardware environment and measured execution times.
Raw data and outputs remain Git-ignored; this report and runner retain the
reviewable experiment specification and results.

Focused checks: 24 tests passed across `test_matter_baseline.py` and
`test_additional_datasets.py`. Tests verify train/evaluation separation,
source-paper overlap detection, exact-claim overlap detection and actual
classifier-to-scorer coverage, alongside existing adapter/scorer checks.
Passing tests do not establish scientific accuracy beyond this measured run.

## Remaining work for an actual system result

1. Acquire historical candidates with the existing date-filtered discovery
   bridge and archive its failures and source provenance.
2. Rank available candidates with the real retrieval/fusion modules, extract
   source-linked passages and preserve dates and exact text.
3. Implement an explicit feasibility decision rule or model. Do not silently
   map three-way NLI output into feasibility labels, and do not use generated
   gold explanations as retrieved evidence.
4. Score complete predictions using the feasibility scorer and compare against
   this diagnostic baseline. Report evidence availability/abstention and timing.
5. If evaluating evidence correctness on this dataset, obtain independently
   validated evidence annotations. Its binary labels alone cannot yield
   evidence precision/recall or paper-ranking relevance judgments.

The historical corpus and feasibility model are still outstanding. This report
closes the missing baseline experiment, not those system integration steps.
