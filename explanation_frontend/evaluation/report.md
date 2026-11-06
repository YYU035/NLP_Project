# Latest integration validation - 6 November 2026

The integrated discovery/retrieval/RRF/abstract-selection/local-NLI/template workflow now passes 113 tests, dependency checks, and separate recorded/fresh-search real-model smoke checks. See end-to-end-smoke.md for measured scope, commands and limitations. The older counts below are historical milestones. The new tests and smoke checks do not supply human evaluation results, evidence-F1 or scientific accuracy claims; optional Qwen generation remains pending.

# Implementation and evaluation status — 31 October 2026

## Free/open-source revision — 4 November 2026

The hosted explanation client was removed at the user's request. The optional LLM path and paired-evaluation mode now load the pinned Apache-2.0 Qwen2.5-0.5B-Instruct model locally with remote code disabled and local-files-only loading. Templates remain the default fallback. Catalogue search explicitly clears provider credentials and retains only anonymous OpenAlex/OpenAIRE/arXiv requests. Earlier references to API configuration below are historical.

The model license, setup instructions and remaining endpoint policy are recorded in API_POLICY.md. Added tests exercise missing/pinned local weights, credential-free configuration, local-only loading, input-limit rejection, and suppression of catalogue credentials.

Validation: **88 tests passed**, dependency/whitespace checks passed, and the 30-case scripted evaluation harness still runs. A fresh independent review found no important API-removal defects. The real local-generation smoke test was **not performed**: the public weight transfer was slow and was stopped while incomplete. No revision marker was activated; incomplete weights remain ignored and cannot be loaded by the app. Run setup-local-llm.ps1 to complete the explicit download before real generation. Templates and cached workflows are usable now.

## Forward implementation — 4 November 2026

The updated suite passes **81 tests**. The app now separates paper discovery, verification and human explanation review. Real recorded discovery uses the exact upstream schema and source revision; it never creates scientific labels. Old version-1 verification exports retain their checksums when optional discovery context is absent.

A live discovery smoke run for the public test claim “Physical exercise improves memory” returned **9 candidates**, **3 HTTP attempts**, and success from OpenAlex/OpenAIRE/arXiv at a per-provider limit of three. The recorded service time was **79.58 seconds**, including its generation/provider work; this single run is not a speed benchmark or quality experiment. Completion was recorded at 2026-11-03T16:09:45Z (4 November, 00:09 Singapore time). The versioned result is saved in ignored outputs/live-discovery-smoke.json. No scientific verdict, PDF verification, OpenAI generation or human study score was produced by that smoke run.

The pinned retrieval dependency and local MiniLM/spaCy models were installed inside this module environment. pip check passes. setup-discovery.ps1 reproduces optional setup without global Git or model-cache changes. The base lock file remains lean; the optional full environment is recorded separately.

AppTest and browser checks covered recorded paper browsing, compact detail/provenance inspection, navigation, the retained verification/citation flow and a blank blinded human-review form. Input state is preserved across workspaces. A fresh code review found a selected-search mode-switch claim mismatch; a regression test reproduced it and the fix restores the exact locked source claim. Human reviews remain genuinely pending; automated form tests do not constitute study reviews.

Source-control work is on local codex/explanation-frontend based on GitHub main. Only explanation_frontend files enter its commits. No upstream branch merge, push or teammate-folder change was performed. Downstream verification factories and the real explanation-quality study remain external integration/evaluation work.

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
