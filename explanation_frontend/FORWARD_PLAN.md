# Forward implementation — 3 November 2026

Goal: fulfil Person 5's explanations, citation checking, Streamlit and evaluation responsibilities using the newly available Person 1 retrieval handoff, without changing teammate folders.

Branch: codex/explanation-frontend, based on GitHub main 8956098. The initial local repository was unborn; fetched refs and a mixed reset established ancestry without checking out teammate files. Missing external files are skip-worktree in the local index only. Commit diffs must remain exclusively under explanation_frontend/. No push is included.

## Decisions

- Keep search discovery distinct from scientific verification. Unranked papers have no scientific verdict until Persons 2–4 provide one. Do not manufacture INSUFFICIENT from absent verification.
- Validate the actual RetrievalResult schema from Raghavi commit 259d2a9. Preserve aliases, multi-provider origins, original records and safe query diagnostics.
- Keep the existing versioned verification/export contract compatible. Add discovery import/export and selection handoff separately.
- Use one persistent asynchronous RetrievalService per live-search session, behind a session-owned bridge and background timeout. Do not repeatedly recreate the service to bypass pacing. Installing optional models is explicit.
- Add a recorded real discovery example with exact upstream source attribution; it is cached and never described as current or scientifically labelled.
- Reorganize navigation into Find papers / Verify evidence / Evaluate explanations. Use compact paginated selection and a neighbouring detail panel; keep verification evidence inspectable and authoritative.
- Add an in-app human review form and report using existing frozen paired evaluation artifacts, preserving explicit pending/failed statuses and real-vs-synthetic distinctions. Do not invent human scores or real scientific findings.

## Tasks and verification

1. [x] Discovery models/import/export: tests for actual saved payload, missing metadata, status distinctions, alias preservation, malformed references, selection tamper and size limits.
2. [x] Live async adapter: tests prove service reuse, original-claim/date/top-k preservation, complete-outage diagnostics and explicit unavailability without synthetic fallback.
3. [x] Search UI and navigation: AppTest confirms cached real papers are Not yet verified, filtering/pagination preserves selected IDs, details/provenance/diagnostics resolve, input edits discard stale live results. Retain existing verification tests.
4. [x] Verification UI: retain two-stage results while reducing scroll; decouple view components from app routing. Preserve test coverage for claim/selection changes, failures, citations and export.
5. [x] Review UI: validate and score blinded raw outputs, export safe review sheets, show per-method denominators and remaining reviews; real model/human work stays pending until performed.
6. [x] Setup/handoff docs: pinned optional discovery installation/model instructions, schema mapping and exact saved-source attribution; record limitations and checks.
7. [x] Full module suite, dependency check, real browser walkthrough, fresh independent review, scope-only commit and branch status verification.

Implementation proceeds autonomously under the user's authorization. All new work and generated artifacts stay within this module; Git metadata changes implement the explicitly requested branch.

## Verification outcome

81 module tests pass. Live discovery smoke: 9 candidates, 3 HTTP attempts, all three providers successful. Browser walkthrough checked discovery, verification tabs/citations and blank human review. A fresh review found the selected-search claim reset and a regression test pins the fix. The optional dependency/model environment is installed locally; actual downstream verification and research human scores remain pending external inputs.
