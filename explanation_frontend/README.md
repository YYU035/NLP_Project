# Explanation frontend - Kireeti, Person 5

The integrated Streamlit workflow now runs end to end: public discovery -> Naila BM25/SBERT -> Yu Yue RRF -> exact abstract-span selection -> Chong Yao local NLI -> cached aggregation -> grounded explanations, paper-subset comparisons and JSON export/replay.

All additions and pinned teammate runtime copies are inside `explanation_frontend/` on branch **kireeti**. Teammate source folders and main are unchanged. See [INTEGRATION_CHANGELOG.md](INTEGRATION_CHANGELOG.md) for every added bridge, policy and source revision.

## Run

On this machine the discovery and NLI models are installed. From the repository root:

```powershell
.\explanation_frontend\run.ps1
```

On a fresh checkout, first install Python 3.12 and run:

```powershell
.\explanation_frontend\setup.ps1
.\explanation_frontend\setup-pipeline.ps1
.\explanation_frontend\run.ps1
```

The setup scripts create the module-local virtual environment, install pinned discovery dependencies and explicitly download public MiniLM/spaCy and MIT NLI model files. Git needs access to the group repository to install Raghavi's pinned paper_search package. The NLI weights are about 870 MB; activation verifies their SHA-256 checksum. Models, environments and artifacts are ignored by Git and remain in this folder. Setup requires internet; model inference does not. No model is downloaded by a verification request.

1. **Find papers:** load the recorded real search, upload a Person 1 RetrievalResult or search export, or request a fresh Live search. Inspect metadata, provider diagnostics, aliases and provenance. Select papers.
2. **Prepare selected papers for verification:** opens Verify evidence with the exact claim locked. Click **Verify claim**. The built-in local pipeline ranks the selected candidates unless a validated ranking is already attached, selects up to three exact adjacent-sentence abstract windows per paper and runs NLI once.
3. Inspect **Verdict & explanation**, **Papers & passages**, and **Run details**. Change paper checkboxes to invoke cached aggregation without neural inference. Export the replayable result.
4. **Cached replay:** load a result without external requests. Original selection retains its saved aggregation; changed replay selections use the explicitly labelled replay policy, not a new call to the live verifier.
5. **Evaluate explanations:** import blinded review packets and record human scores. Export review CSV/JSON before closing the session.

Alternatively, Verify evidence -> **Live** performs discovery and the remaining stages in one request, using ten candidates per provider. Search/ranking scores and NLI confidence are not scientific truth probabilities. A cancelled or timed-out request produces no inferred verdict; late results are discarded. The initial pipeline deadline is 300 seconds, allowing for CPU inference and cold model loading; subset aggregation has a five-second UI deadline. This is a local capstone app, not a production multi-user service.

## Person-by-person integration checklist - 6 November 2026

| Owner / boundary | Current status | Remaining owner work |
| --- | --- | --- |
| Person 1 - Raghavi | **[CONNECTED]** Pinned discovery package, live and recorded searches, canonical IDs, exact claim, aliases, provider diagnostics, source records and selected-paper lineage. Fresh public search exercised with the downstream pipeline. | **[PENDING - Person 1 + team]** Full-text acquisition, retraction/freshness/source-version handling and source eligibility agreement. PDF links are metadata links only. |
| Person 2 - Naila | **[CONNECTED]** Pinned BM25/SBERT runtime, native Paper adapter, local MiniLM encoder, real ordered branch lists and branch query timings. No title/abstract means no retrieval score; the paper remains visible as an unavailable verification input. | **[PENDING - Naila + team]** Frozen benchmark results and model-quality comparison. Integration smoke is not retrieval-quality evaluation. |
| Person 3 - Yu Yue | **[CONNECTED]** Pinned RRF runtime consumes actual Person 2 branch rankings, validates canonical identities and retains ranked records/scores in exports. Manual RankingInput/RankingOutput import remains available. | **[PENDING - Yu Yue + team]** Optional cross-encoder numerical/dependency validation. RRF-only is the supported default. |
| Passage selection | **[ADDED BY KIREETI]** Exact, deterministic abstract/description windows with original character offsets, stable passage IDs and lexical selection. No PDF is fetched or claimed to have been read. | **[PENDING - Persons 3-4 + team]** Confirm owner and replace/approve this fallback. Preserve dataset sentence indices in any future SciFact evidence-selector evaluation. |
| Person 4 - Chong Yao | **[CONNECTED]** Pinned schemas/classifier, offline NLI loading, typed response mapping, original-response caching and actual cached aggregate method. Real local inference, conflict handling, subsets and export/replay exercised. | **[PENDING - Person 4 + team]** Review the documented frontend conflict policy and run broader held-out calibration/quality/latency checks. |
| Person 5 - Kireeti | **[CONNECTED]** Built-in factory, discovery and verification UI, grounded template explanations/citations, scoped subset results, operational failures, export/replay and human-review workspace. | **[PENDING - Kireeti + reviewers]** Real human explanation/citation evaluation. Optional local Qwen setup/paired comparison is not required for the working template pipeline and remains pending. |
| Whole team | **[WORKING END TO END, ABSTRACT ONLY]** Fresh public discovery and recorded-search pipelines passed real local-model smoke checks. | **[PENDING - all owners]** Team acceptance of added policies, broader edge-case and scientific quality evaluation, full-text support, report/demo preparation. |

## What was verified

**113 tests passed** and **pip check passed**. The automated suite covers contracts, identities, original claims, ranking provenance, exact spans, truncation/operational failures, internal/across-paper conflicts, cached subset aggregation, Streamlit navigation and export/replay. The latest run and reproducible smoke commands are recorded in [evaluation/end-to-end-smoke.md](evaluation/end-to-end-smoke.md).

Real CPU smoke checks on 6 November 2026:

- Recorded real search: 32 candidates, 3 selected papers, 9 passages, valid grounded explanation and replay; approximately 28.20 seconds including local model loading; first subset aggregation about 49.98 ms.
- Fresh public search: 9 candidates, 3 selected papers, 9 passages, valid grounded explanation and replay; approximately 31.44 seconds including discovery/model loading; first subset aggregation about 10.43 ms.

These are functionality checks on two claims, not accuracy, scientific correctness or quality benchmarks. The selector uses provider abstract/description fields, whose authenticity and interpretation are not independently verified.

## Validation commands

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
.\explanation_frontend\.venv\Scripts\python.exe -m pytest explanation_frontend/tests -q -o cache_dir=explanation_frontend/.pytest_cache --basetemp=explanation_frontend/.tmp/pytest
.\explanation_frontend\.venv\Scripts\python.exe -m pip check
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.pipeline_smoke
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.pipeline_smoke --live --limit 3
```

The recorded smoke makes no external requests. To prove inference is offline, set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before it. The live smoke still explicitly calls public discovery APIs. Reports and replay files go to ignored `outputs/integration/`. Regression tests use synthetic classifier/encoder doubles for pipeline orchestration; the smoke commands use real models.

## Explanation policy and optional local LLM

Templates quote supplied eligible passages and preserve the verifier's verdict. Truncated, neutral, changed or out-of-selection passages cannot become primary explanation citations. Internal conflicts and opposing papers retain eligible evidence from both sides. No source text is executable, and a replay checksum is not source authentication.

Optional Qwen/Qwen2.5-0.5B-Instruct generation is local, Apache-2.0 and bounded by the same extractive gate. Run setup-local-llm.ps1 explicitly to install it. Its previous download remains incomplete; template explanations are the tested default. Missing/unsupported local generation falls back visibly to templates. Scientific usefulness still requires human evaluation.

The evaluation harness, review rubric and existing synthetic cases remain available under `evaluation/`. Synthetic acceptance rates must not be reported as real LLM or scientific quality results. See [evaluation/rubric.md](evaluation/rubric.md).

## Configuration and handoffs

The built-in factory is `explanation_frontend.pipeline:create_provider`; no environment factory configuration is needed. Trusted server overrides EXPLANATION_PIPELINE_FACTORY and EXPLANATION_VERIFY_DISCOVERY_FACTORY remain supported. They are not upload-supplied code paths. The `.env.example` file documents settings and is not automatically loaded.

- [INTEGRATION_CHANGELOG.md](INTEGRATION_CHANGELOG.md): source revisions, complete additions log, policy changes and limitations.
- [RANKING_INTEGRATION.md](RANKING_INTEGRATION.md): manual ranking imports and contract.
- [PERSON4_INTEGRATION.md](PERSON4_INTEGRATION.md): native response and cached aggregator mapping.
- [contracts.md](contracts.md): local snapshots, citation lineage and factory interfaces.
- [API_POLICY.md](API_POLICY.md): no hosted/proprietary inference or paid API dependency.

Free anonymous public discovery APIs are the only claim-search network calls. Credentials are cleared. All embeddings, ranking, passage selection, verification, aggregation and explanations run locally. Provider access/rate-limit failures remain explicit; no paid fallback is used. Full-text reading, arbitrary PDF upload, independent scientific verification and production hosting are outside this implemented abstract pipeline.
