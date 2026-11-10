# Full-corpus ranking and evidence accuracy experiment

## Result

This is a completed real-model experiment on the fixed SciFact corpus. Train configuration selection used 200 deterministically sampled training claims; all 300 development claims were evaluated. Development data was already inspected and is not an untouched test set.

The training objective selected `bounded_abstract` at depth five. All predeclared policy/depth ablations are reported; development performance did not select that policy.

| Policy | Papers verified | Accuracy | Macro-F1 | Positive-only accuracy | Evidence F1 | Complete rationale document coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 0.5667 | 0.5839 | 0.5053 | 0.1756 | 136/209 |
| baseline | 10 | 0.5400 | 0.5623 | 0.5106 | 0.1273 | 141/209 |
| baseline | 20 | 0.5267 | 0.5545 | 0.5000 | 0.0892 | 146/209 |
| neighbors_0.6 | 5 | 0.6067 | 0.6228 | 0.5585 | 0.1839 | 143/209 |
| neighbors_0.6 | 10 | 0.5767 | 0.5993 | 0.5638 | 0.1341 | 148/209 |
| neighbors_0.6 | 20 | 0.5633 | 0.5919 | 0.5532 | 0.0948 | 153/209 |
| bounded_abstract | 5 | 0.6733 | 0.6781 | 0.6649 | 0.0949 | 168/209 |
| bounded_abstract | 10 | 0.6567 | 0.6683 | 0.6596 | 0.0638 | 179/209 |
| bounded_abstract | 20 | 0.6367 | 0.6523 | 0.6489 | 0.0419 | 185/209 |

Train-selected depth-five accuracy change versus baseline: 0.10666666666666667. Descriptive paired claim bootstrap 95% interval: [np.float64(0.043333333333333335), np.float64(0.16666666666666666)].

Paired correctness transitions: `{'correct -> correct': 140, 'wrong -> correct': 62, 'wrong -> wrong': 68, 'correct -> wrong': 30}`.

## Scope and interpretation

RRF uses BM25 Porter stemming and dense MiniLM retrieval, full 5183-paper corpus, top100 per branch, RRF k=60. No gold papers are injected. The training sample is the first 200 SHA-256-ordered claim IDs, independent of labels. All train claims, including those without positive evidence, participate in the accuracy objective.

Baseline: three relevance-ranked sentence segments at cosine >=0.5. Selective context: add immediate neighbors with cosine >=0.6. Bounded abstract: if the same relevance gate finds any baseline anchor, use the exact whole abstract when the evidence-claim pair fits 512 tokenizer tokens; otherwise retain baseline passages. No text truncation or gold-based passage choice. Thresholds and policies were fixed before this run.

The fixed pretrained verifier, threshold 0.5 and owner/frontend aggregation are unchanged. MPS float32 results are recorded separately from CPU. The actual passage predictions are cached by exact claim and text, then re-aggregated at Top-5/10/20. Per-paper verification is independent of other papers; therefore prefix re-aggregation does not require repeated neural inference. No per-condition latency comparison is inferred from cached execution.

Longer passages may improve verdict decisions while lowering complete-sentence evidence precision by including unannotated material. Gold absence is benchmark INSUFFICIENT, not a universal scientific verdict. Correctness and evidence quality must be reported separately.

## Verification

All 2700 development condition rows are present and unique. Accuracy and Macro-F1 were independently recomputed with scikit-learn for every condition. Model/data hashes, numerical environment, training IDs, prediction cache accounting and source provenance are recorded in the exported protocol. Compare `baseline-agreement.json` with the original frontend run; reconstruction is explicitly checked rather than assumed.

Full passage-level traces and reusable neural predictions remain under `outputs/ranking_evaluation/accuracy-context-full-batched`. Compact train/dev case rows, results and every changed verdict are exported under `accuracy-full-results/`. No models or vector arrays are included in the handoff.

This report does not modify production configuration or frontend behavior. Any adoption should use the train-selected depth-five policy and preserve these baseline results; an independent test is still needed for a generalization claim.
