# Evidence selection case review

## Reproduction and scope

Run `python ranking_evaluation/analysis/kireeti_validation_20261110/inspect_selection.py` from the repository root. The audit verifies both SciFact source hashes against the frozen evaluation, matches archived selected passages uniquely to original source segments, and writes `selection_cases.json` with original sentences, gold indices, production segments, selected text and recorded NLI predictions.

This is a two-case diagnostic on previously inspected development claims. The context variant expands the archived anchors by one production segment on each side without using gold labels to choose anchors or neighbors. Gold labels are used only to measure coverage. It does not rerun the encoder, enforce the verifier token budget, rerun NLI, or measure new verdict accuracy. It does not change production code or frozen results.

## Claim 137: missed results and conclusion

Claim: “Asymptomatic visual impairment screening in elderly populations does not lead to improved vision.”

Gold paper: 26016929. Gold label: SUPPORTED. Annotated evidence: original sentences 7 and 9.

- Sentence 7 reports that screening did not improve self-reported visual problems, with the pooled odds ratio and confidence interval.
- Sentence 9 concludes that screening asymptomatic older people is not justified on present evidence.
- Actual selected original sentences: 0 (study objective), 8 (a small reduction cannot be excluded), and 10 (visual impairment can usually be reduced with treatment).
- All three gold-paper passages were predicted NEUTRAL. Both the gold-paper-selector and retrieved-paper routes return INSUFFICIENT; the annotated-rationale condition returns SUPPORTED.

The gold evidence sentences survive the production segmentation intact but are not selected. For this case, changing sentence boundaries alone would not recover the missing evidence. The relevance selector favors related wording without preserving the study's answer/conclusion.

Expanding each archived anchor by one adjacent production segment covers both gold sentences. Selected segments increase from three to seven. This is evidence-coverage recovery, not proof that the verifier will now predict correctly.

## Claim 1163: missed evidence plus downstream scope error

Claim: “The DdrB protein from Deinococcus radiodurans is an alternative SSB.”

Gold paper: 15305881. Gold label: SUPPORTED. Annotated evidence: original sentence 5, describing DdrB's distinct fold compared with other SSB proteins.

- Actual selected original sentence coverage: 0, 2, 3 and 6. The selected production segment containing original sentences 2 and 3 merges across `spp. that`; this demonstrates that original dataset sentence boundaries and live string segmentation differ. It does not remove the unselected gold sentence 5, which remains an intact production segment.
- All three gold-paper selected passages are predicted NEUTRAL.
- Even the annotated-rationale condition returns INSUFFICIENT. Supplying the gold evidence therefore does not resolve this case with the frozen verifier.
- In the full route, paper 799586 supplies a passage about alternative SSB in Bacillus subtilis and Streptococcus pneumoniae. Its recorded NLI label is CONTRADICT with confidence 0.6478974. This is the only CONTRADICT label among the archived passages, and the full route returns CONTRADICTED. The different species/protein scope needs manual review with Person 4.

One-segment context expansion covers gold sentence 5, but increases selected segments from three to seven and covers all eight original abstract sentences. This is a substantial context increase and cannot be treated as a free improvement. A bounded variant needs token-budget checks, latency measurements and an actual verifier rerun.

## Decision for the next experiment

Use a separately versioned, bounded context variant as the first candidate, retaining the frozen three-anchor selector as baseline. Context support already exists through `SelectionConfig.context_radius`; a new implementation should only be added if existing behavior cannot support the experiment. Do not overwrite the train-frozen configuration or tune thresholds on these two examples.

Before reporting an improvement, compare the baseline and candidate on a predefined development protocol, count token-budget rejections, and report evidence precision/recall/F1, complete-rationale coverage and actual verdict accuracy/Macro-F1. More context can also add distracting or contradictory material. Keep the existing full-sentence scorer, with clearly labeled supplemental overlap diagnostics if needed.

Claim 137 is the clearer selector case. Claim 1163 must also be assigned to verifier/scope review. Both cases have their gold papers in the evaluated candidate scope, so neither is evidence that increasing paper depth alone will fix the error.

## Checks completed

Both frozen source hashes matched. All six selected gold-paper passages mapped uniquely to the original abstract segments and exact text. Both baseline coverage reconstructions matched the archived absence of complete gold rationale. Both context coverage diagnostics executed successfully. No model inference, frontend changes, model parameter changes, or new push was performed.
