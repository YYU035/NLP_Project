# Audit of Kireeti's frozen full-pipeline validation

Kireeti's branch through `51f5408` was merged into local `yuyue` without conflicts. This is an artifact audit, not a new model run. Original frontend code, frozen configuration, and reported predictions were preserved.

Reproduce from repository root:

```bash
python ranking_evaluation/analysis/kireeti_validation_20261110/analyze.py
```

The script checks condition accounting, recomputes counts and paired transitions, records input SHA-256 hashes in `diagnostics.json`, and extracts all 15 available claim replays into `replay_cases.json`. The archived replays are a selected diagnostic sample, not all 300 full traces. The CSV contains all 676 condition-level outcomes.

## Accuracy comparison on matched claims

| Condition | Correct / same 188 annotated-positive claims | Accuracy |
| --- | --- | --- |
| Annotated rationale supplied | 115 / 188 | 61.2% |
| Correct papers supplied; production selector | 100 / 188 | 53.2% |
| Retrieved top-five papers; production selector | 95 / 188 | 50.5% |

Full-route accuracy across all 300 claims is separately 170/300 = 56.7%. This includes 112 claims without annotated positive evidence. Such benchmark negatives do not establish that no supporting paper exists elsewhere.

The first transition loses 23 previously correct predictions and gains eight (net loss 15). The second loses nine and gains four (net loss five). These are paired descriptive changes, not independent causal contributions: evidence inputs, extra papers, NLI outputs, and aggregation interact. Oracle accuracy is not a mathematical performance ceiling.

## Person 3: ranking and evidence selection

There are 209 annotated claim-document pairs across the 188 positive claims. The top-five route includes 181/209 (86.6%). This is pooled annotated-document coverage, not mean query Recall@5, and does not establish how many additional correct papers rank sixth or lower.

With all correct papers supplied, selected spans cover a complete annotated rationale for 150/209 document pairs (71.8%). In the retrieved route, this falls to 136/209 (65.1%). Complete-rationale coverage therefore deserves investigation even when paper retrieval succeeds. These are document-level figures, not claim accuracy.

No spans were rejected by the verifier token budget in these positive-claim conditions. Increasing the token limit would not recover rejected spans in this frozen run, although selecting more context could change future input lengths.

The evidence scorer requires coverage of whole original annotated sentences. It records five partial sentence fragments in the gold-paper condition and 20 in the retrieved condition; these counts include fragments outside gold sentences. Partial coverage receives no full-sentence credit and contributes a false positive. This custom conservative scorer is not the official SciFact scorer. Unannotated relevant sentences can also be counted as false positives. The evidence F1 values need this qualification, but the accuracy failures cannot all be dismissed as scoring artifacts.

The top-five restriction is explicit in `explanation_frontend/evaluation/validation/scifact.py`. It is an offline experiment wrapper; it does not prove the UI always restricts users to five papers.

## Inspectable cases

- **Claim 137 — selector/context review.** Gold SUPPORTED; full route INSUFFICIENT. A gold paper is retrieved but no complete annotated rationale is covered. Selected text includes the study objective, a possible small reduction in reported visual problems, and a generic treatment statement. Another paper contributes a fragment ending `cognitive impairment (78.4 vs.`. This supports checking objective-versus-conclusion selection and sentence boundaries around abbreviations/numbers. It does not prove which individual sentence caused the final failure.
- **Claim 113 — aggregation/verification review.** Gold SUPPORTED; full route CONFLICTING despite a retrieved gold paper and complete rationale coverage. Paper 6157837 is classified SUPPORTED and contains the passage saying ACE inhibitor use can be associated with functional renal insufficiency. Paper 14584755 is classified CONTRADICTED with confidence about 0.9895. Opposing paper predictions trigger conflict. Review the contradicting passage's scope with Person 4 before changing aggregation; suppressing disagreement blindly would hide legitimate conflicts.
- **Claim 1163 — joint selection/verifier review.** Gold SUPPORTED; full route CONTRADICTED. The gold paper is present but complete annotated rationale coverage is absent. Selected passages discuss DdrB binding/function and another species' alternative SSB. Species and protein identity need checking; the trace alone cannot attribute the error exclusively to ranking or NLI.
- **Claims 3, 5, 436 and 528 — downstream review.** Each has a retrieved gold paper and complete rationale coverage, yet the final verdict is wrong. Missing gold papers or incomplete rationale coverage alone cannot explain these cases. Extra passages, NLI and aggregation still require inspection.

Full selected passages, paper-level verdicts, passage predictions and processing failures are preserved in `replay_cases.json` and linked original replays.

## Next experiments

1. Recover the complete frozen ranking lists and all per-case traces from the evaluation host. Compare top-5/10/20 using the same ranked lists and report document recall, complete-rationale coverage, evidence precision/recall/F1, verdict accuracy/Macro-F1 and actual latency. More papers can improve recall and introduce conflicting predictions; do not assume accuracy improves.
2. Audit sentence boundaries against the original SciFact sentence arrays, then evaluate a separately versioned selector that retains conclusion/context sentences. Report both strict complete-sentence and explicitly labeled overlap diagnostics. Preserve the original scorer for comparisons.
3. Coordinate with Person 4 on wrong oracle-rationale predictions and scope/negation errors; coordinate with Kireeti on additional-paper conflicts. Hold ranking and selection fixed for downstream comparisons.
4. Choose changes using a separate development protocol; these already inspected SciFact dev cases cannot establish unseen-test improvement.

The merged frontend already includes XAI components. This does not imply all explainability checks passed: its final report flags Integrated Gradients convergence failures. Three-dataset retrieval evaluation and SciFact labeled evidence/verdict evaluation have different annotation coverage; CSFCube and RECIPER do not acquire SciFact-style claim-verdict or sentence-rationale labels from this merge.

## Validation limits

The audit script ran successfully against the merged artifacts and checked 300 full-route rows plus two sets of 188 controls. No new NLI inference, model accuracy experiment, frontend test suite, or GPU run was performed for this audit. Original model-result correctness remains subject to the frozen study's protocol and documented limitations. No model parameters were changed.
