# Implementation and evaluation status — 31 October 2026

## What was executed

The module test suite currently passes 52 tests under Python 3.12.15, Streamlit 1.58.0 and Pydantic 2.12.5. It exercises snapshot linkage, selection isolation, explanations, malformed/unavailable generations, HTTP-independent LLM injection, export/replay, request identity, timeout handling, per-session state, UI flows, live subset background/caching behavior, the injected integration adapter and evaluation summaries.

The Streamlit server was started locally and its real browser flow exercised: conflict scenario → Verify claim → uncheck paper 2 → selected SUPPORTED while the provisional result remains CONFLICTING. The UI visibly labels the synthetic source and displays the exact cited passage. A local preview is saved at outputs/app-preview.png.

A fresh reviewer identified large-evidence rendering, preservation of the typed verifier response and synchronous subset aggregation issues. Regression tests reproduce them; the implementation now permits all template evidence, retains the original response, and caches subset background work with a five-second UI deadline. Additional hardening rejects tampered limitation prose, non-finite retrieval scores and decisive verdicts without matching decisive evidence.

## Synthetic harness result (not model-quality evidence)

Command: `python -m explanation_frontend.evaluation.run --mode fixture`.

| Item | Observed |
| --- | --- |
| Frozen synthetic stress cases | 30 |
| Paired outputs | 60 |
| Scripted accepted generations | 22 |
| Scripted rejected generations | 7 |
| Scripted unavailable generation | 1 |
| Displayed template fallbacks | 8 |
| Completed human reviews | 0 |
| Pending human review rows | 120 (two reviewers × 60 outputs) |

The scripted failures include altered numbers, fabricated citations, verdict disagreement, invalid JSON and a timeout. These are intentional test inputs, not estimates of real-model failure rates. All cases use invented papers/findings; none is represented as SciFact. No real API or NLI inference was performed for this harness.

## Research findings still pending

- Real LLM smoke run and paired generation on the team's frozen held-out evidence snapshots.
- Two independent human reviews (or a documented single-reviewer limitation), with disagreement/adjudication and per-method support/usefulness scores.
- Quantitative semantic citation support, supported-correction rates, usefulness and real provider latency/cost. No such values are claimed here.
- Real cached outputs from all four upstream modules and live end-to-end integration. The integration-adapter test proves the call boundary and no repeated inference for subsets using injected functions; it does not prove the teammates' implementations work.
- Real aggregate latency and scientific verification behavior. The replay aggregator is explicitly a local policy, not a validation of Person 4's scientific rules.

The current LLM presentation gate accepts exact whole passages only, so template and accepted LLM explanations may look very similar. This is an intentional initial baseline. Human review must still assess whether chosen passages actually support the explanation and remain useful, and compare raw rejected output separately from displayed fallback.
