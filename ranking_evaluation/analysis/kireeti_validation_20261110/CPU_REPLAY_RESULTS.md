# Local CPU verifier comparison

Actual pinned DeBERTa inference on all 15 prepared replay claims, with both baseline and candidate rerun on CPU float32. Semantic selection inputs were prepared using MPS float32; this is a CPU verifier run, not an entirely CPU preprocessing experiment.

These are selected diagnostic cases enriched for failures. Results cannot be generalized to the full 300-claim development set or an unseen test set.

| Condition | Accuracy | Macro-F1 | Positive-only accuracy | Verifier/aggregation seconds | Operational failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.2000 | 0.1333 | 0.0000 | 22.67 | 0 |
| neighbors_0.6 | 0.2667 | 0.2035 | 0.1111 | 20.04 | 0 |

Matched-case transitions: `{'correct -> correct': 3, 'wrong -> correct': 1, 'wrong -> wrong': 11}`.

New CPU baseline verdicts matching the archived predictions: 15/15.

## Verdict changes

- Claim 1274: gold SUPPORTED; INSUFFICIENT -> SUPPORTED. The tip of the inner tube of the toxic type VI secretion system (T6SS) antibacterial effector in Escherichia coli (E. coli) carries toxic effector proteins.

## Interpretation and reproducibility

Evidence scores alone cannot establish verdict improvement. Inspect the actual paired transitions above before deciding whether to change any configuration. No production configuration or frontend was changed.

Latency includes actual verification and aggregation for each condition, without cross-condition prediction caching; excludes network download, model loading, retrieval, selection preparation and frontend rendering. CPU warmup/order effects are not controlled, so these are operational timings, not a rigorous latency benchmark. Claims are processed baseline then candidate.

The model weight checksum, device, dtype, package versions, prepared input hash and condition metrics are recorded in `summary.json`. Full paper and passage predictions are in `predictions.jsonl`. The prepared input retains the original claims, paper scope, source evidence text and tokenizer budget checks.

Reproduce from repository root with a new output directory using `pipeline_context.py infer --device cpu`, the existing prepared JSONL and pinned local NLI directory. No model threshold, weights, claim text or evidence labels were changed.


## Validation completed

All 30 condition rows completed without operational failures. Independent accuracy accounting agreed with the summary, and all recorded passage confidences and raw probabilities were finite. Sixteen related selector, budget and ranking validation tests passed, with one optional integration skipped. The downloaded 870,182,606-byte weight file matched the fixed SHA-256 before model loading. Full 300-claim evaluation remains pending.
