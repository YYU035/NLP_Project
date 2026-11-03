# Explanation frontend — Person 5

An independently runnable Streamlit workspace for inspecting scientific claim verification. It includes template explanations, an optional guarded OpenAI explanation generator, passage-level citations, paper selection and verdict changes, processing states, JSON export/replay, and paired explanation-evaluation tools.

All implementation, configuration, fixtures, tests, environments and generated files stay inside this directory. No upstream module has been implemented or changed here.

## Run now

The local Python 3.12 virtual environment is already installed. From the project root in PowerShell:

```powershell
.\explanation_frontend\run.ps1
```

Open [the local app](http://127.0.0.1:8501). Stop it with Ctrl+C. The launcher binds to your machine only and disables Streamlit usage telemetry. If port 8501 is occupied, use `-Port 8502`.

Choose Demo → conflict → Verify claim. Uncheck paper 2: the provisional result remains CONFLICTING, while the selected result becomes SUPPORTED. Uncheck all: the selected result becomes INSUFFICIENT with EMPTY_SELECTION. Change the claim: the prior result disappears. The failure scenario shows PROCESSING FAILURE, not a scientific verdict. All demo papers/claims/findings are synthetic.

Demo only accepts the scenario's fixed claim; it cannot verify arbitrary scientific text. Use cached real outputs or the configured live bridge for real claims.

## Install on another machine

Python 3.12 is tested. Supply its executable explicitly if your system's default Python differs:

```powershell
.\explanation_frontend\setup.ps1 -Python 'C:\path\to\python.exe'
.\explanation_frontend\run.ps1
```

The installer creates .venv, pip cache and temporary files locally. requirements.txt pins direct dependencies; requirements-lock.txt records the tested Windows/Python 3.12 environment and transitive versions. Other platforms can create a virtual environment and install requirements.txt; their resolved dependency versions should be recorded separately.

## Optional LLM explanations

Templates work without credentials. Set these in the terminal that starts the app:

```powershell
$env:OPENAI_API_KEY = '<your key>'
$env:EXPLANATION_LLM_MODEL = '<a model available to your account with Chat Completions JSON output>'
$env:EXPLANATION_LLM_TIMEOUT = '20'
.\explanation_frontend\run.ps1
```

.env.example documents the variables; .env files are deliberately not auto-loaded. Do not commit your key. The client uses OpenAI Chat Completions with JSON output, 3,000 completion tokens, a bounded per-attempt timeout and at most one retry for HTTP 429/5xx. Model compatibility and account access must be verified with your chosen model. No specific model is silently chosen.

Click Generate LLM explanation only when you want to send the exact claim and selected evidence to OpenAI. Checkbox changes do not automatically make LLM calls. Generated statements must be WHOLE exact supplied passages, with valid passage IDs and matching verdict. Free-form paraphrases, missing qualifiers, invented citations and unsupported corrections fall back visibly to the template. Prompt injection is constrained by the output gate; arbitrary source markup is displayed as text.

This deliberately conservative first version is extractive. It does not independently establish semantic relevance or scientific truth. The evaluation harness retains rejected raw model output for human assessment; the normal app never renders that raw output.

Corrections are withheld unless the upstream snapshot explicitly supplies approved replacement evidence. Merely finding a different number in a passage is insufficient. See contracts.md before enabling this upstream feature.

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

To run the paired real experiment, create a frozen-case manifest (see evaluation/rubric.md), configure the model and run:

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

Real upstream wiring awaits the four teammates' functions and agreed payload mappings. Real-model smoke testing requires API configuration. Scientific explanation-quality conclusions require frozen real cases and actual human reviews. The supplied detailed downloadable edge-case checklist was unavailable; the grouped 27 Person 5 IDs are covered in evaluation/edge-cases.md without inventing individual priorities. General paper-search/QA features, spreadsheet export, arbitrary PDF upload and independent scientific verification are outside this module.

API references used: [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing), [Streamlit fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment), [OpenAI Chat API](https://developers.openai.com/api/reference/resources/chat).
