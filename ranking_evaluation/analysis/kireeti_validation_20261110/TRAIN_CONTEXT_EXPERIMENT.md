# Train-selected selective context experiment

## Decision

Retain the deployed frozen baseline. Keep `neighbors_0.6` as an experimental candidate: it improves development evidence coverage slightly, but its micro-F1 change is small and the descriptive uncertainty interval includes zero. No new verdict accuracy has been measured.

## Protocol

Use the same 200 SHA-256-ordered training claim IDs as the earlier selector study, independent of evidence labels. Of these, 139 have annotated evidence and 61 have no gold-paper selection inputs. Use all 300 development claims; 188 have annotated evidence, and 112 have no gold-paper inputs. Train and dev claim IDs are disjoint. Inspect `train-context/protocol.json` for exact claim-text overlap and source hashes; ID separation does not rule out related claims or shared papers.

Supply gold evidence papers for both splits, explicitly isolating sentence selection from retrieval. Join abstract sentences as production strings and use the existing production segmentation heuristic. Compute all relevance scores with the pinned MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, max sequence length 256, MPS float32 on the local Mac GPU. Verify local model file hashes before encoding. Model weights remain unchanged.

Fix top-three anchors and cosine threshold 0.5. Predeclare six policies: baseline, unconditional neighboring context, and immediate neighbors gated by cosine 0.3, 0.4, 0.5 or 0.6. Anchor and neighbor choices use only text relevance. Merge overlapping windows while preserving exact source text. Use the actual pinned DeBERTa tokenizer to reject whole windows exceeding 512 evidence-claim pair tokens, without truncation or fallback. The train objective maximizes positive-only micro evidence F1, with predefined tie-breaks. Freeze the selected policy before development scoring; all six predeclared development ablations are retained without changing that choice.

The training objective selected `neighbors_0.6`: train F1 0.42991 versus baseline 0.42938. This very small train difference is part of the result, not evidence of a robust improvement.

## Development results: correct papers supplied

| Policy | Precision | Recall | F1 | Complete rationale document pairs |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.3670 | 0.5355 | 0.4356 | 150/209 |
| All immediate neighbors | 0.2433 | 0.7486 | 0.3673 | 179/209 |
| Neighbor score at least 0.3 | 0.2663 | 0.7131 | 0.3878 | 173/209 |
| Neighbor score at least 0.4 | 0.2888 | 0.6913 | 0.4074 | 171/209 |
| Neighbor score at least 0.5 | 0.3255 | 0.6421 | 0.4320 | 165/209 |
| Train-selected neighbor score at least 0.6 | 0.3530 | 0.5874 | 0.4410 | 157/209 |

No policy produced over-budget windows in this experiment. Accepted token-pair totals were 30,502 for baseline and 30,843 for the train-selected candidate; accepted passage counts were 532 and 406 respectively. Overlap merging explains why fewer passages can contain more text. Token counts are not measured verifier latency.

The selected policy's micro-F1 change is +0.00547 (about +0.55 percentage points). A paired claim bootstrap, 2,000 repeats with seed 20261110, gives a descriptive 95% percentile interval of [-0.01075, +0.02200]. This includes zero and does not model related claim/document dependence. Do not claim statistically established improvement or independent test performance.

## Validation and reproducibility

- All 188 baseline claim-level evidence scores and complete-rationale counts exactly match Kireeti's archived gold-paper condition.
- All selected spans preserve the exact abstract substring. Both train and dev use the actual verifier tokenizer budget checks.
- Policy behavior, baseline equivalence, source offsets, neighbor gating, overlap merging and existing selector/budget interfaces: 15 tests passed, one optional integration test skipped.
- Neural vectors, source data, model files, tokenizer files, package versions and experiment script provenance are recorded in the protocol. Vectors are cached locally and not committed as model artifacts.

From repository root:

```bash
.venv/bin/python ranking_evaluation/analysis/kireeti_validation_20261110/train_context.py --device mps
.venv/bin/python ranking_evaluation/analysis/kireeti_validation_20261110/check_train_context.py
PYTHONPATH=live_search/src:. .venv/bin/python -m pytest ranking_evaluation/analysis/kireeti_validation_20261110/test_train_context.py ranking_evaluation/tests/test_semantic_evidence.py ranking_evaluation/tests/test_verifier_budget.py -q
```

The runner requires the existing pinned encoder and tokenizer cache directories. The CPU option uses float64 and is a separately recorded numerical protocol. Reusing cached vectors requires identical text ordering, model revision, device and dtype. Do not mix CPU and MPS result claims.

## Remaining evaluation

This is an actual semantic selection experiment, unlike the earlier archived-anchor replay. It still supplies the correct papers: it is not a top-five retrieved-paper run. No new NLI inference, Accuracy, Macro-F1, inference latency or end-to-end latency was measured. Development data has already been inspected and should not be described as an untouched test set. The original complete-sentence scorer remains unchanged and can penalize plausible unannotated context.

A full-pipeline comparison should preserve this train-frozen choice, use actual retrieved candidates, load the exact pinned verifier weights on the evaluation host, and compare new verdicts against baseline. Until that comparison exists, keep the candidate in the experiment directory and leave the frontend and train-frozen production configuration unchanged.
