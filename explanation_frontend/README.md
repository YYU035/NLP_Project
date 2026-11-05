# Explanation frontend — Person 5

An independently runnable Streamlit workspace for inspecting scientific claim verification. It includes template explanations, an optional local Apache-2.0 explanation model, passage-level citations, paper selection and verdict changes, processing states, JSON export/replay, and paired explanation-evaluation tools.

All implementation, configuration, fixtures, tests, environments and generated files stay inside this directory. No upstream module has been implemented or changed here.

## New workflow — 4 November 2026

The sidebar now separates **Find papers**, **Verify evidence**, and **Evaluate explanations**.

- **Find papers:** load the recorded real search (32 candidates), upload a native Person 1 RetrievalResult/discovery export, or perform live discovery. Browse ten papers per page beside an inspectable detail panel. Query/provider diagnostics, aliases, all origins and original normalized records remain available. Filtering does not erase selections. All candidates are **Not yet verified**.
- **Verify evidence:** the existing template/LLM and subset workflow is split into Verdict & explanation / Papers & passages / Run details tabs. Prepare a frozen paper selection directly from discovery; ranking and verification are still the downstream owners' responsibility.
- **Evaluate explanations:** upload blinded review_packets.jsonl and submit explicit human scores against exact evidence. Reviewer ID, usefulness rating and acknowledgement are required. Export scores as CSV/JSON before closing the session. Unreviewed outputs remain pending; the synthetic practice set is clearly labelled.

The optional retrieval package and pinned local MiniLM/spaCy models are installed in this local environment. A live smoke run returned **9 candidates, 3 HTTP attempts, all three providers successful** for the public test claim “Physical exercise improves memory”, with a per-provider limit of three. It generated no scientific verdict. Its result is saved in ignored outputs/live-discovery-smoke.json, as a smoke check rather than a retrieval-quality experiment.

The recorded sample is copied byte-for-byte from live_search/artifacts/depth-approved-run/c01.json at Raghavi's upstream commit `259d2a9e90fb8e9d5988f030a156733f4f3576d1`. It is cached, contains no verification labels, and does not imply a new search or endorsement of its claim. The native payload supplies no retrieval timestamp, so the app leaves that unknown.

### Live discovery setup on another machine

After normal setup:

```powershell
.\explanation_frontend\setup-discovery.ps1
.\explanation_frontend\run.ps1
```

This installs requirements-discovery-lock.txt, including the private GitHub package at the reviewed revision. Git must have access to the group repository. It explicitly downloads MiniLM revision `2a7cca6ad7605f3183bce07f677bda658059bbe9` into .models/all-MiniLM-L6-v2; dependencies include en_core_web_sm 3.8.0. Weights/caches remain ignored inside this module. Use -SkipModels only when pinned weights are already present.

Choose Find papers → Live search. This frontend intentionally clears all provider credentials and uses only the free, anonymous catalogue-search paths; configured keys are ignored. One async service/event loop per session preserves provider pacing and query cache across requests; arXiv fallback stays off. Thirty means candidates **per provider**, not a globally ranked top 30. The UI distinguishes partial results, successful empty search and complete outage. The bridge has a 115-second deadline and the UI a 120-second bound. Cancellation discards the result; abandoned work must finish before a new search. Release live search connection closes the HTTP resources.

### Selected-search verification

Prepare selected papers for verification moves a frozen selection into Verify evidence → Selected search and locks its original claim. When Persons 2–4 are ready, configure:

```powershell
$env:EXPLANATION_VERIFY_DISCOVERY_FACTORY = 'explanation_frontend.your_bridge:create_provider'
```

The trusted factory returns a provider exposing run_from_discovery(discovery_snapshot, selected_IDs) and aggregate(verification_snapshot, selected_IDs). FromDiscoveryProvider supplies the guarded seam using injected verify_selected(handoff) and aggregate(snapshot, IDs) functions. The former is the owners' ranking → passage selection → verification pipeline returning a mapped RunSnapshot; the latter reuses its cached response.

Returned snapshots must preserve the exact claim and a subset of selected canonical IDs. The adapter attaches the full original discovery payload/selection to optional RunSnapshot.discovery_context, preserving aliases, origins and cached/live provenance. No search is repeated. Existing version-1 verification exports remain replayable: absent discovery context does not change their hashes. The actual downstream factory remains unconfigured until the owners supply their functions.

## Run now

The local Python 3.12 virtual environment is already installed. From the project root in PowerShell:

```powershell
.\explanation_frontend\run.ps1
```

Open [the local app](http://127.0.0.1:8501). Stop it with Ctrl+C. The launcher binds to your machine only and disables Streamlit usage telemetry. If port 8501 is occupied, use `-Port 8502`.

Choose Verify evidence → Demo → conflict → Verify claim. Under Papers & passages, uncheck paper 2: the provisional result remains CONFLICTING, while the selected result becomes SUPPORTED. Uncheck all: the selected result becomes INSUFFICIENT with EMPTY_SELECTION. Change the claim: the prior result disappears. The failure scenario shows PROCESSING FAILURE, not a scientific verdict. All demo papers/claims/findings are synthetic.

Demo only accepts the scenario's fixed claim; it cannot verify arbitrary scientific text. Use cached real outputs or the configured live bridge for real claims.

## Install on another machine

Python 3.12 is tested. Supply its executable explicitly if your system's default Python differs:

```powershell
.\explanation_frontend\setup.ps1 -Python 'C:\path\to\python.exe'
.\explanation_frontend\run.ps1
```

The installer creates .venv, pip cache and temporary files locally. requirements.txt pins direct dependencies; requirements-lock.txt records the tested Windows/Python 3.12 environment and transitive versions. Other platforms can create a virtual environment and install requirements.txt; their resolved dependency versions should be recorded separately.

## Optional local open-source explanations

The paid/proprietary hosted explanation client has been removed. Templates require no model and work fully offline. The optional LLM is **Qwen/Qwen2.5-0.5B-Instruct**, released under **Apache-2.0**, run directly on this computer with the open-source PyTorch and Transformers libraries. No hosted inference endpoint, API key, subscription or paid account is used.

The local model is pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`. Download it explicitly once:

```powershell
.\explanation_frontend\setup-local-llm.ps1
.\explanation_frontend\run.ps1
```

This installs requirements-local-llm.txt and downloads publicly available weights/tokenizer/license files into this module's ignored .models directory. The download uses the public model repository without authentication. After setup, inference loads local files only with remote code disabled. Claim/evidence text stays on this computer. Runtime never downloads a missing model or switches to cloud inference.

The weight file is approximately 988 MB. Its size and SHA-256 checksum are checked before activation. The attempted download on this machine was stopped while incomplete because it was slow; run the setup command to finish installation before using real local generation. Templates work without it. No real local-model quality or inference result is claimed yet.

The installed model directory is the default; optional EXPLANATION_LOCAL_MODEL_PATH selects a local copy of the same pinned revision, and EXPLANATION_LOCAL_TIMEOUT controls the generation time limit (default 60 seconds, maximum 120). .env.example documents these settings; no .env file is auto-loaded. CPU compute, memory and electricity are local operating costs; there are no per-request inference charges.

Generate local LLM explanation uses deterministic greedy decoding, a bounded token budget and no silent input truncation. The existing grounding gate remains: statements must be whole supplied passages with valid IDs and a matching verdict; unsupported corrections/qualifier changes/fabricated citations fail visibly to the template. The small CPU model may produce malformed or unsupported output, which is retained only for evaluation and withheld from normal display. No quality advantage over templates is assumed.

See API_POLICY.md for the remaining free public discovery endpoints and explicit setup-only downloads.

## Cached real results and live integration

Download replayable result creates a self-contained JSON bundle with exact claim, full supplied evidence, metadata/versions, selection, verdicts, model settings, failures and final explanation. Cached replay → Load replay validates it without network calls. It preserves original provenance while showing CACHED REPLAY. Changed replay selections use the documented local replay policy; unchanged selections retain the saved verdict exactly. A checksum is not a signature of authenticity.

The live bridge is dependency-injected because teammates' modules are not available in this checkout. Implement a trusted factory in your own bridge file inside explanation_frontend, using IntegratedProvider and the mapping in contracts.md. Then configure:

```powershell
$env:EXPLANATION_PIPELINE_FACTORY = 'explanation_frontend.your_bridge:create_provider'
```

The factory returns a new provider instance for the session/run. Its initial run calls search → rank → select → verify once. Changed subsets call the cached original verification response through Person 4's aggregator, run in a background job and cached by snapshot/selection. The UI has a five-second aggregate deadline and a 90-second initial-pipeline deadline; upstream calls must also have their own timeouts. The target for actual aggregate computation remains well under one second.

Cancellation abandons the result; it cannot forcibly kill Python/network/model work. Late results are discarded. At most one remote/background request per session is admitted; if abandoned work is still running, new remote requests wait for it to finish. This prevents duplicate paid calls and unbounded worker creation. Stateless clients are not used to share users' mutable results. Production authentication, distributed job queues and multi-process deployment are outside this local capstone build.

## Verify and evaluate

From the project root:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\explanation_frontend\.venv\Scripts\python.exe -m pytest explanation_frontend/tests -q -o cache_dir=explanation_frontend/.pytest_cache --basetemp=explanation_frontend/.tmp/pytest
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.evaluation.run --mode fixture
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.evaluation.run --reviews explanation_frontend/outputs/evaluation/review.csv
```

The synthetic harness produces 60 blinded packets for 30 stress cases, a private method key, raw/displayed records, automatic status summary, and blank sheets for two reviewers. It is NOT a real LLM experiment. Do not use its scripted acceptance rates as evidence of model performance.

To run the paired real experiment, create a frozen-case manifest (see evaluation/rubric.md), install the local open-source model and run:

```powershell
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.evaluation.run --mode llm --cases explanation_frontend/outputs/heldout-cases.jsonl --out explanation_frontend/outputs/real-study
```

Human reviewers score factual support, citation support/completeness, verdict agreement, scope/number/uncertainty preservation, supported corrections and usefulness. The summary joins the private method key and reports per-method denominators and paired reviewer verdict agreement. No missing human score is filled automatically. Record final split/model/prompt versions before evaluation and keep development tuning separate. Treat raw generation quality and safe displayed fallback quality as different outcomes.

## Files to hand off

- models.py and contracts.md: exact local handoff schema and owner questions.
- providers.py: fixture provider, replay policy and injected real integration adapter.
- templates.py, llm.py, validation.py: explanation generation and display gate.
- app.py, state.py, export.py: UI, request identity, bounded work and JSON replay.
- fixtures/: 12 reproducible synthetic scenarios; regenerate with `python explanation_frontend/fixtures/build.py`.
- tests/: automated contracts, runtime, UI, integration-adapter, hardening and evaluation tests.
- evaluation/: 30-case synthetic manifest, reviewer rubric, current report and edge-case ledger.
- outputs/: ignored local review packets, test artifacts and preview; do not commit private real claims/evidence by accident.

## Current limits and remaining external work

Person 1 discovery and Yu Yue's RRF runtime are connected through pinned implementations. Real Person 2 branch rankings and downstream evidence/verification functions and agreed mappings are still pending. Explanation generation is local; evaluation of its scientific usefulness still requires human review. Scientific explanation-quality conclusions require frozen real cases and actual human reviews. The supplied detailed downloadable edge-case checklist was unavailable; the grouped 27 Person 5 IDs are covered in evaluation/edge-cases.md without inventing individual priorities. General paper-search/QA features, spreadsheet export, arbitrary PDF upload and independent scientific verification are outside this module.

API references used: [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing), [Streamlit fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment), [Qwen model/license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

## Connected ranking workflow

The paper browser now accepts Person 2 `RankingInput` or Yu Yue RRF-only `RankingOutput`, runs the pinned local RRF implementation, displays ranks/scores and retains the output through selected-paper verification handoffs and replay exports. See [RANKING_INTEGRATION.md](RANKING_INTEGRATION.md) for steps and remaining owner dependencies. No model/API is called by RRF; actual BM25/SBERT results must be supplied.

## Person-by-person integration checklist (5 November 2026)

Owner: **Kireeti, Person 5**. Published branch: **kireeti**. This checklist distinguishes implemented frontend connections from real teammate outputs that have not yet been supplied or validated. The checked GitHub heads remain main `8956098`, Raghavi `259d2a9`, and Yu Yue `a91e257`. This branch changes only `explanation_frontend/` relative to main; teammates' modules remain on their branches or optional dependencies.

| Integration | Connected and checked in Person 5 | Remaining task and responsible owner |
| --- | --- | --- |
| Person 1 — Raghavi: discovery → candidate browser | Native RetrievalResult import, pinned live discovery bridge, recorded real search, exact claim, canonical IDs/aliases, provider/query provenance, metadata/PDF links, selection, export and handoff | **[PENDING — Person 1 + Kireeti]** Validate full-text availability, freshness/retraction metadata and agreed source/version handling on real team outputs. PDF links alone are not inspected full text. |
| Person 2 — Naila: BM25/SBERT → ranking input | Downloadable RankingInput contract; import accepts unique ordered canonical IDs and branch timings; exact candidate/claim matching; unknown/duplicate IDs rejected | **[PENDING — Person 2]** Supply runnable BM25/SBERT retrieval and real ordered lists. **[PENDING — Person 2 + Kireeti]** Connect its callable and test automatic retrieval → RRF on real claims. Empty downloaded lists are placeholders. |
| Person 3 — Yu Yue: RRF → ranked browser | Pinned unchanged RRF runtime runs locally; RankingInput/RankingOutput import, recomputation checks, final/branch rank and score display, ranked selection, export/replay and ranked downstream handoff | **[PENDING — Yu Yue + Kireeti]** Validate optional cross-encoder environment and its reported CPU NaN issue before enabling it; current integration accepts RRF-only results. **[PENDING — Persons 2–3 + Kireeti]** Validate rankings using actual Person 2 outputs. |
| Evidence passage selection — ownership to confirm | Frontend passage/evidence schemas, paper/passage identity checks, scoped citation rendering and fixtures exist | **[PENDING — Persons 3–4 + team]** Confirm who owns passage selection: the task split and new ranking README assign it differently. **[PENDING — evidence owner + Kireeti]** Map real extracted passage text, IDs, sections, sentence offsets, source versions and truncation flags into the frontend. |
| Person 4 — Chong Yao: verifier/aggregator → verdict/explanation | IntegratedProvider and FromDiscoveryProvider seams; ranked selected-paper handoff; original claim/identity checks; cached subset aggregation interface; asynchronous jobs, deadlines and stale-result protection | **[PENDING — Person 4]** Supply runnable passage verification and cached subset aggregation. **[PENDING — Person 4 + Kireeti]** Implement the real factory/mapping, configure EXPLANATION_VERIFY_DISCOVERY_FACTORY or EXPLANATION_PIPELINE_FACTORY, and validate thresholds, conflicts, failures and subset latency. No fixture result counts as this integration. |
| Person 5 — Kireeti: evidence → explanation/UI/evaluation | Streamlit discovery/verification/review workspaces, grounded templates, citation/verdict gates, conflict and insufficient-evidence displays, snapshot export/replay, review packets and evaluation harness | **[PENDING — Kireeti]** Finish optional local Qwen download and verify actual inference if using the LLM comparison. **[PENDING — Kireeti + reviewers]** Freeze real held-out cases, collect human reviews and report real explanation/citation quality. Templates already work offline. |
| Whole team: full chain | Individual boundaries and synthetic/recorded regression workflows are available | **[PENDING — all owners + Kireeti]** Run real discovery → BM25/SBERT → RRF → passage selection → verification/aggregation → explanations, including conflict, empty/partial failure, truncation and changed selections. Confirm strict downstream parsers accept the additive ranking_context handoff field. |

### Execution checklist

- [x] Connect Person 1 candidate discovery and preserve source provenance.
- [x] Connect Yu Yue's RRF implementation without an API or model call.
- [x] Preserve ranking context and final order through selection, exports and verification handoffs.
- [x] Provide frontend contracts, adapters and explicit missing-integration errors.
- [x] Validate the current implementation: **100 tests passed**; **pip check passed** in the local development environment. Ranking-specific tests require the optional paper_search dependency.
- [ ] **[PENDING — Person 2]** Deliver real BM25/SBERT results and retrieval callable.
- [ ] **[PENDING — Persons 3–4]** Confirm passage owner and deliver real extracted evidence.
- [ ] **[PENDING — Person 4]** Deliver verifier and cached aggregator outputs/callables.
- [ ] **[PENDING — Kireeti + module owners]** Configure real factories and run the automatic full-chain acceptance checks.
- [ ] **[PENDING — Kireeti, optional]** Complete and smoke-test local Qwen inference; validate cross-encoder with Yu Yue separately.
- [ ] **[PENDING — Kireeti + human reviewers]** Complete real-case explanation evaluation and document results.

**Completion assessment:** the standalone frontend and discovery/RRF boundaries are implemented and regression-tested. Full scientific pipeline integration is incomplete until real retrieval, passage and verification outputs pass the shared workflow. An overall percentage would conflate working frontend features with missing external modules, so the owner-specific statuses above are the completion measure.

Start integration using [RANKING_INTEGRATION.md](RANKING_INTEGRATION.md); use [contracts.md](contracts.md) for verification/evidence mappings and [API_POLICY.md](API_POLICY.md) for the free public discovery and local-only inference policy. No proprietary or paid inference API is required by Person 5.
