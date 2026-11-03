# Implementation ledger

Scope: every created or modified file remains inside explanation_frontend/.

Ruling: execute the accepted plan in this workspace without Git/worktree mutations, because the user expressly limited writes to this directory. No shared schema or teammate files will be changed.

Ruling: use installed Python 3.12 for a module-local virtual environment; system Python 3.15 is not the compatibility target.

Ruling: real upstream integration and human explanation scores cannot be fabricated. Deliver an injected integration adapter, runnable fixture journey, paired-evaluation tooling and explicit pending evidence for those external dependencies.

Ruling: use whole, exact passages for displayable LLM evidence statements. A matching substring can drop negation or qualifications. Free-form paraphrases are retained only in raw evaluation artifacts and rejected for user display. Corrections require upstream explicit replacement evidence; none is inferred locally.

Progress and verification results are recorded below as implementation proceeds.

## Delivered

- Task 1: strict versioned handoff schemas, reference/score checks, 12 synthetic fixtures and a provider protocol.
- Task 2: whole-passage template baseline, source/citation inspection and explicit insufficiency/failure/conflict states.
- Task 3: configurable OpenAI JSON generation, conservative gate, raw-output evaluation retention and visible template fallback. Real provider smoke check blocked: API key/model not configured.
- Task 4: Streamlit two-stage flow, persistent paper selection, background live work, cancellation/late-result handling and isolated sessions.
- Task 5: validated self-contained JSON export/replay; no spreadsheet export. Replay selection policy is explicitly labelled.
- Task 6: paired/blinded harness, 30 synthetic stress cases, reviewer rubric, raw/displayed records, per-method summaries and pending human review sheets. Real evidence study and human scores are pending, not fabricated.
- Task 7: injected integration adapter preserves the original typed verifier response and proves subset changes do not repeat inference in tests. Real upstream functions remain externally unavailable; no other module was edited.

## Review and verification

Fresh read-only code review found large-evidence template limits, typed-response mismatch and synchronous repeated subset aggregation. Regression tests failed before fixes; templates now support larger sets, the original response is retained, and live subsets use cached background jobs with a five-second deadline.

The new async UI test exposed widget state resetting during fragment/full reruns. Separate persistent selection state now restores the widget values and preserves subset identity across completion reruns.

Additional regression tests reject altered limitation text, non-finite retrieval scores and decisive verdicts without matching evidence. Invalid raw generations have no fabricated validation result; displayed fallback validation is recorded separately.

Verified: 52 tests pass; pip check reports no broken requirements; PowerShell launcher/installer parse; local Streamlit health endpoint returns 200 ok. Browser verification exercised conflict→supported by deselecting a paper; screenshot is outputs/app-preview.png. Synthetic evaluation: 30 cases, 60 outputs, 22 scripted accepted / 7 rejected / 1 unavailable, 8 visible fallbacks, zero completed human reviews.

Ruling: leave the app running on loopback port 8501 for the user's review. No commits, pushes, root configuration edits or changes outside explanation_frontend were made. The pre-existing docs/ plan remains unchanged.
