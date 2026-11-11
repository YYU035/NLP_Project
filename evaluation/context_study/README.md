# Research runner: context and contradiction protection

This runner tests changes without altering the production pipeline, original baseline, model weights or NLI threshold. The original full-corpus baseline must already exist under outputs/validation/scifact_full (created by validate.cmd).

Run context-study.cmd from explanation_frontend. It uses the module-local environment and pinned local models. It retains completed frozen studies instead of silently selecting another policy after seeing results. Interrupted runs reuse exact-input NLI cache entries. Training records refuse overwrite.

The stages are:

1. freeze_train: use the pinned owner's actual 200-claim training predictions to select a predeclared prediction-only guard. Maximize training accuracy and macro F1 subject to retaining baseline contradiction recall.
2. train_gate: apply the guard; select a context paper gate from 0.50/0.55/0.60/0.65/0.70 using the same training objective. Protected decisive predictions bypass the gate. Original SciFact training bytes come from the official public archive and must match the owner's recorded SHA-256.
3. run_local: compare five predeclared arms on all 300 dev claims using the fixed full-corpus RRF top five. Reuse the measured original CPU focused predictions; run bounded-abstract inputs through the actual pinned local NLI model. No gold paper injection or claim rewriting.
4. evaluate_gate: apply the separately frozen training gate to the completed native traces. No new inference is required for subset aggregation.
5. report: independently check all result accounting, compute paired differences, retain all regressions and export charts, CSVs and replay examples.

Guarded context evidence metrics count the union of focused input and retained context. The saved replay displays the decisive arm; complete auxiliary input traces remain under outputs/context_study_20261111/local_traces. Longer context is not represented as a small highlighted anchor for scoring.

Training/dev IDs are disjoint but one exact claim text overlaps; an additional text-disjoint sensitivity check is reported. These development labels were already inspected, so none of this establishes unseen-test performance or a universal accuracy improvement. Related papers/statements may also overlap.

Read REPORT.md for measured results, results/changed_claims.csv for every changed candidate verdict, and results/contradiction_regressions.csv for the previous candidate's lost contradictions. The review ZIP is outputs/context_study_20261111/review-kit.zip. No human review scores are invented.
