# Bounded context replay experiment

## Result and decision

Keep the frozen baseline. On this selected diagnostic sample, adding one neighboring production segment on either side improves evidence recall but reduces evidence F1. No production configuration was changed.

| Metric | Baseline radius 0 | Context radius 1 |
| --- | ---: | ---: |
| Annotated-positive claims | 9 | 9 |
| Gold evidence sentences recovered | 7 / 16 | 12 / 16 |
| Evidence precision | 0.1094 | 0.0889 |
| Evidence recall | 0.4375 | 0.7500 |
| Evidence F1 | 0.1750 | 0.1589 |
| Gold document pairs with a complete rationale | 5 / 9 | 8 / 9 |
| Over-budget spans, all 15 claims | 0 | 0 |
| Accepted passages, all 15 claims | 106 | 72 |

Passage counts fall because overlapping context windows merge; fewer passages do not imply less text. Precision and F1 use the original conservative complete-sentence scorer, which can penalize relevant but unannotated context. These numbers do not establish that the extra context is factually irrelevant.

## Protocol

Use all 15 archived claim replays, including nine annotated-positive claims and six without annotated positive evidence. This is a selected, nonrandom case sample enriched for failures, not the full 300-claim development run. Keep the archived paper scope and accepted relevance anchors fixed. Reuse the existing selector's overlap merging, top-three anchors and 0.5 relevance threshold. Compare radius zero with radius one. Gold labels never choose anchors or expansion boundaries.

For every proposed passage, check the actual pinned DeBERTa tokenizer with evidence as premise, original claim as hypothesis, special tokens included, no truncation, and a 512-token pair limit. Reject an over-budget merged window whole without fallback. Only tokenizer files were downloaded; no verifier model weights or new neural predictions were produced.

Baseline reproduction checks passed for all 15 cases: accepted passage text matched the archived baseline and evidence scores matched exactly. Both SciFact source hashes matched the frozen evaluation. The output records source and tokenizer file hashes, individual passages, token counts and per-claim evidence scores.

## Reproduction

From the repository root, with the pinned tokenizer already provisioned in `.cache/ranking/verifier-tokenizer`:

```bash
.venv/bin/python ranking_evaluation/analysis/kireeti_validation_20261110/context_experiment.py
PYTHONPATH=live_search/src:. .venv/bin/python -m pytest ranking_evaluation/tests/test_semantic_evidence.py ranking_evaluation/tests/test_verifier_budget.py -q
```

The tokenizer checkpoint and exact revision are recorded in `context_results.json` under `tokenizer_contract`. The environment requires the existing frontend evaluation and transformers dependencies. The runner is offline and does not silently download models. Output: `context_results.json`.

Related selector and budget tests: 12 passed, one skipped (optional provisioned-tokenizer integration). The experiment itself exercised the actual downloaded verifier tokenizer. The first test invocation lacked the live-search source import path; rerunning with the repository's source path above passed.

## Limits and next decision

There is no new Accuracy, Macro-F1, verifier latency or end-to-end latency measurement. Archived verdicts cannot be reused as predictions for changed evidence. Per-case selection/tokenization timings are diagnostic overhead only. No full model run or full-development claim is implied.

The result supports a more selective context policy, rather than blanket deployment of radius one. Any such policy should be fixed using train data before a broader development comparison, with unchanged baseline retained. A full model comparison also needs the exact pinned NLI weights and complete evaluation-host artifacts; the checked-in archive provides only 15 full traces. The 676-row CSV supports outcome accounting but cannot reconstruct missing passage traces or ranking lists.
