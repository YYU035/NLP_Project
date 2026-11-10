# Train-selected paper gate and verifier context

## Result

The policy and paper threshold were selected using 200 training claims before applying this comparison to development labels. The chosen policy is `bounded_abstract`, paper relevance threshold 0.65, fixed primary scope Top-5.

All 300 SciFact development claims are included. This is inspected development data, not a new unseen test.

| Condition | Top-K candidate scope | Accuracy | Macro-F1 | Positive-only accuracy | Evidence precision | Evidence recall | Evidence F1 | Complete rationale pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 0.5667 | 0.5839 | 0.5053 | 0.1073 | 0.4836 | 0.1756 | 136/209 |
| baseline | 10 | 0.5400 | 0.5623 | 0.5106 | 0.0728 | 0.5055 | 0.1273 | 141/209 |
| baseline | 20 | 0.5267 | 0.5545 | 0.5000 | 0.0488 | 0.5246 | 0.0892 | 146/209 |
| train_selected | 5 | 0.6700 | 0.6663 | 0.5798 | 0.0928 | 0.6120 | 0.1612 | 138/209 |
| train_selected | 10 | 0.6600 | 0.6575 | 0.5745 | 0.0804 | 0.6639 | 0.1434 | 148/209 |
| train_selected | 20 | 0.6567 | 0.6548 | 0.5745 | 0.0693 | 0.6803 | 0.1258 | 152/209 |

Primary Top-5 accuracy delta: +0.1033; descriptive paired bootstrap interval [np.float64(0.04), np.float64(0.17)].

Paired correctness transitions: `{'correct -> correct': 136, 'wrong -> correct': 65, 'wrong -> wrong': 65, 'correct -> wrong': 34}`.

## What changed

Keep the same RRF rankings and pretrained verifier. Whole-abstract context is permitted only after the existing relevance gate and only when the exact premise/claim pair fits 512 tokens; long abstracts fall back to original selected spans. An additional training-selected paper gate removes papers whose maximum accepted-passage relevance is below the frozen threshold. No new papers are injected, no source text or claim is rewritten, no model weights or NLI threshold changed. The same owner/frontend aggregation runs on the retained paper scope.

The gate uses cosine relevance, not a calibrated truth probability. It trades evidence coverage for fewer weakly matching papers. Full-abstract context also includes unannotated sentences: evidence precision/F1 can decrease even when verdict accuracy rises. The table reports actual verifier-input evidence scores; it does not hide extra context by scoring only the short anchors.

Top-10/20 results are ablations, not development-selected deployment choices. The primary candidate remains the train-frozen Top-5 configuration.

## Case review

- Claim 137: gold SUPPORTED; baseline INSUFFICIENT; candidate SUPPORTED.
- Claim 1163: gold SUPPORTED; baseline CONTRADICTED; candidate SUPPORTED.
- Claim 113: gold SUPPORTED; baseline CONFLICTING; candidate SUPPORTED.
- Claim 1274: gold SUPPORTED; baseline INSUFFICIENT; candidate SUPPORTED.

Every changed verdict, including regressions, is recorded in `relevance-gate-results/paired-comparison.json`. Full input/output traces are retained in the source evaluation run.

## Backend use

The reusable adapter is `ranking_evaluation.contextual_evidence.BoundedAbstractSelector`. Load `ranking_evaluation/experiments/contextual-accuracy-train-frozen.json`, wrap the original configured selector and actual verifier tokenizer, and explicitly pass `min_paper_score`. Persist `context_configuration` with the output. The default selector/frontend was not changed. See `ranking_evaluation/CONTEXTUAL_EVIDENCE.md`.

## Verification and limits

All 1800 condition rows are unique and complete. Accuracy and Macro-F1 were independently verified with scikit-learn. Training trace hashes are frozen and checked before dev replay. Filtering/reaggregation uses real cached NLI predictions: changing paper scope never changes the individual evidence-premise/original-claim inputs. No extra inference is required to remove papers.

Related claims or papers may cross train/dev IDs. The descriptive paired bootstrap is not a guarantee of independent-test gains. Source benchmark negatives reflect absent annotations, not a universal claim about all literature. These experiments evaluate SciFact verdicts; the other two datasets do not provide equivalent verdict labels.

No frontend modifications or git push were performed. Independent test evaluation and teammate integration remain separate tasks.


## Per-class tradeoffs and final interface verification

At Top-5, SUPPORTED recall increased from 47/124 (37.9%) to 75/124 (60.5%), while INSUFFICIENT recall increased from 75/112 (67.0%) to 92/112 (82.1%). CONTRADICTED recall decreased from 48/64 (75.0%) to 34/64 (53.1%). This is an overall Accuracy/Macro-F1 improvement, not improvement for every class. It requires particular care if missing contradictions has higher application cost.

The reusable backend adapter matched evaluated accepted passages on all 7,000 train/dev claim-paper scopes. All 10,952 cached neural predictions had finite confidences and raw probabilities. The 300 original baseline verdicts and evidence scores matched the original frontend run exactly. Person 3 tests: 170 passed, one skipped. The original frozen selector and frontend defaults remain unchanged; the new configuration is opt-in.
