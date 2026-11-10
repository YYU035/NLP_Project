# Accuracy improvement handoff

Completed real-model evaluation on all 300 SciFact development claims, with full 5183-paper retrieval. The selected backend candidate improves overall Accuracy from 170/300 (56.7%) to 201/300 (67.0%) and Macro-F1 from 0.584 to 0.666. The existing baseline was exactly reproduced on all 300 verdicts and evidence scores.

The candidate retains relevant whole-abstract context within the actual verifier's 512-token pair budget, falls back to baseline spans for long abstracts, and abstains on weak papers using a train-selected cosine gate of 0.65. The semantic encoder, verifier weights, NLI threshold and aggregation rules are unchanged. All policy/gate choices used 200 training claims. Development data was already inspected: these are development results, not independent test claims.

## Ranking conclusion

| Original sentence selector | Accuracy |
| --- | ---: |
| RRF Top-5 | 56.7% |
| RRF Top-10 | 54.0% |
| RRF Top-20 | 52.7% |

Increasing paper depth raised retrieval recall but lowered verdict accuracy. Keep the train-frozen primary candidate at Top-5. Its Accuracy at Top-10 and Top-20 is 66.0% and 65.7%, respectively; these are ablations, not development-selected defaults.

## Evidence and error tradeoffs

Evidence recall increased from 48.4% to 61.2%, but full-sentence evidence precision fell from 10.7% to 9.3% and F1 from 0.176 to 0.161 because longer context includes more unannotated sentences. Complete annotated rationale coverage is 136/209 versus 138/209. Do not claim all evidence metrics improved.

The candidate fixes 65 original errors and introduces 34 regressions, net 31 more correct claims. In the reviewed cases 113, 137, 1163 and 1274, the candidate now predicts the correct SUPPORTED label. CONTRADICTED recall decreases from 75.0% to 53.1%; SUPPORTED and INSUFFICIENT recall increase. Keep this class tradeoff visible when deciding whether to integrate.

## Backend integration

- Implementation: `ranking_evaluation/contextual_evidence.py`, `BoundedAbstractSelector`.
- Frozen opt-in settings: `ranking_evaluation/experiments/contextual-accuracy-train-frozen.json`.
- Usage and metadata: `ranking_evaluation/CONTEXTUAL_EVIDENCE.md`.
- Full metrics, every regression, uncertainty and source hashes: `RELEVANCE_GATE_RESULTS.md` and `relevance-gate-results/`.
- Three-policy, three-depth ablations: `FULL_ACCURACY_RESULTS.md` and `accuracy-full-results/`.
- Full local passage traces/cache: `outputs/ranking_evaluation/accuracy-context-full-batched/`.

The adapter's accepted passages match all 7,000 evaluated train/dev claim-paper scopes. All 10,952 neural predictions have finite confidences/probabilities. Person 3 test suite: 170 passed, one skipped. Batched model-forward validation matched 197/197 sequential passage labels; its maximum confidence difference was 0.0000027. Device, batching, seed-cache and numerical details are recorded.

The original defaults and frontend were not modified. Kireeti can opt into the new backend selector using the frozen settings and retain the original baseline for comparisons. Model files and vector caches remain local and ignored by Git. No git push was performed.

## Reproduction

With the pinned encoder and verifier caches provisioned, run from repository root:

```bash
.venv/bin/python ranking_evaluation/analysis/kireeti_validation_20261110/accuracy_experiment.py \
  --output outputs/ranking_evaluation/accuracy-context-reproduction \
  --encoder-device mps --nli-device mps --batch-size 4

.venv/bin/python ranking_evaluation/analysis/kireeti_validation_20261110/report_accuracy_experiment.py \
  --run outputs/ranking_evaluation/accuracy-context-reproduction
```

The paper-gate train command writes an immutable choice and refuses to overwrite an existing choice. Keep the checked-in frozen gate for reproduction of this candidate. `relevance_gate_experiment.py dev --run ...` verifies all recorded training trace hashes before applying that choice. A from-scratch new run must create its own frozen record/output location if numerical or hardware changes alter trace hashes; do not overwrite the existing study. Full-corpus model evaluation is offline, separate from live API paper discovery. CSFCube and RECIPER retrieval studies remain unchanged; they do not supply these claim-verdict labels.
