# Full-pipeline validation — 10 November 2026

The full automated offline route completed all **300 SciFact dev claims** with **zero processing failures**, **56.7% claim accuracy** and **0.584 macro F1**. This is a frozen descriptive audit of previously inspected dev data, not an untouched test set. Retrieval is stronger than final verification: RRF Recall@10 is 95.6% on eligible SciFact claims, 39.2% on CSFCube test and 93.7% on RECIPER. These are different retrieval tasks and protocols.

Read the [complete test report](evaluation/validation/FINAL_TEST_REPORT.md), [per-condition outputs](evaluation/validation/results/test_results.csv), [error catalogue](evaluation/validation/results/error_catalogue.csv) and [machine-readable metrics](evaluation/validation/results/validation-summary.json). Representative real-model replays are in `evaluation/validation/results/replays`; the local review ZIP is `outputs/validation/review-kit.zip`. Open Project overview for report/error downloads. Run `validate.cmd` to reproduce checks and reuse completed frozen studies.

Clean setup: 163 frontend tests pass; all 452 pinned owner tests pass. Seven PDF cases, 30 retrieval repeat checks and five repeated NLI pairs pass. Synthetic scientific-language expectations pass 12/15; population/generalisation errors remain. IG convergence is still unresolved. **Software passes do not imply high scientific accuracy.** Older release notes below describe their original narrower scopes.

# Current demo release — 9 November 2026

Start with `start.cmd` and open **Project overview**. Run `test.cmd` for the complete test suite and dependency check; it creates its own module-local temporary directory. See [FINAL_INTEGRATION.md](FINAL_INTEGRATION.md) for the current integration status, actual measurements and additions. [DEMO_RUBRIC_GUIDE.md](DEMO_RUBRIC_GUIDE.md) provides the five-person recording script and proof for efficiency, reusability, presentation and participation. Historical notes below are superseded where these documents state newer results.

Word occlusion is now the default model explanation; Naila’s IG remains experimental. Eight actual-model runs, explicit replays, three-domain charts, measured zero-inference subset changes and actual-output review packets are ready. Real video participation and human scores still require the team.

XAI validation update (9 November): baseline-aware attribution, stricter numerical gates, reproducible SciFact evaluation and a full-abstract verifier audit are documented in [XAI_VALIDATION.md](XAI_VALIDATION.md). Initial PAD failure is historical; MASK128 converges on the synthetic example, the four real SciFact pilot cases failed at128 steps, and a targeted512-step case narrowly failed the conservative absolute gate. Real-case reliability remains measured per input.

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

The setup scripts create the module-local virtual environment, install pinned discovery dependencies and explicitly download public MiniLM/spaCy, the exact Person 3 semantic encoder and MIT NLI model files; optional local PDF/Captum dependencies are installed too. Git needs access to the group repository to install Raghavi's pinned paper_search package. The NLI weights are about 870 MB; activation verifies their SHA-256 checksum. Models, environments and artifacts are ignored by Git and remain in this folder. Setup requires internet; model inference does not. No model is downloaded by a verification request.

1. **Find papers:** load the recorded real search, upload a Person 1 RetrievalResult or search export, or request a fresh Live search. Inspect metadata, provider diagnostics, aliases and provenance. Select papers.
2. **Prepare selected papers for verification:** opens Verify evidence with the exact claim locked. Click **Verify claim**. The built-in local pipeline ranks the selected candidates unless a validated ranking is already attached, uses Yu Yue’s train-frozen semantic selector for up to three exact abstract passages per paper, rejects whole spans exceeding the verifier token budget and runs NLI once.
3. Inspect **Verdict & explanation**, **Papers & passages**, and **Run details**. Change paper checkboxes to invoke cached aggregation without neural inference. Export the replayable result.
4. **Cached replay:** load a result without external requests. Original selection retains its saved aggregation; changed replay selections use the explicitly labelled replay policy, not a new call to the live verifier.
5. **Local PDF:** upload a text PDF for bounded, page-aware local verification and replay. **Benchmark results:** inspect three domain-specific published evaluations. **Model attribution:** inspect diagnostics for a frozen passage; unstable importance scores are withheld.
6. **Evaluate explanations:** import blinded review packets and record human scores. Export review CSV/JSON before closing the session.

Alternatively, Verify evidence -> **Live** performs discovery and the remaining stages in one request, using ten candidates per provider. Search/ranking scores and NLI confidence are not scientific truth probabilities. A cancelled or timed-out request produces no inferred verdict; late results are discarded. The initial pipeline deadline is 300 seconds, allowing for CPU inference and cold model loading; subset aggregation has a five-second UI deadline. This is a local capstone app, not a production multi-user service.

## Person-by-person integration checklist - 9 November 2026

| Owner | Connected and validated | Remaining work |
| --- | --- | --- |
| Person 1 - Raghavi | Public discovery, canonical identities, selected-paper provenance and fresh search feeding actual local inference. | **[PENDING - Person 1 + team]** Automated full-text acquisition, retraction/freshness and source eligibility review. Local uploaded PDFs are now supported separately. |
| Person 2 - Naila | BM25/SBERT retrieval; local integrated-gradients workspace and 12 attribution plumbing tests. | **[PENDING - Naila + team]** Complete broader real-model attribution convergence and usefulness studies; inspect the new gated pilot/audit in XAI_VALIDATION.md. Unstable token highlights are withheld. |
| Person 3 - Yu Yue | Updated RRF, train-frozen semantic selector, exact source spans, tokenizer budget and local PDF extraction. Published SciFact, CSFCube and RECIPER results displayed with provenance and separate protocols. Current isolated owner suite: 163 passed. | **[PENDING - Yu Yue + team]** Independently reproduce full published benchmarks, review held-out protocol and broaden real PDF/layout/evidence quality studies. |
| Person 4 - Chong Yao | Local NLI, native response mapping, cached aggregation, successful selector abstention, conflict safety, page-aware evidence and replay. | **[PENDING - Person 4 + team]** Calibration/held-out verdict quality and acceptance of documented frontend aggregation policy. |
| Person 5 - Kireeti | Six connected workspaces, grounded explanations/citations, diagnostic and failure handling, three-dataset reporting, PDF/attribution UI, subset comparisons and export/replay. Frontend suite: 136 passed. | **[PENDING - Kireeti + reviewers]** Human explanation/citation and PDF reviews; Qwen remains experimental after its earlier schema failures. |
| Whole team | Recorded and fresh discovery -> retrieval -> ranking -> semantic evidence -> actual NLI -> explanation -> replay exercised. Synthetic local PDF also exercised with actual models. | **[PENDING - all owners]** Scientific quality sign-off, report/demo, and confirmation that Materials Science satisfies the requested third domain if Physics is specifically required. |

See [COMBINED_INTEGRATION.md](COMBINED_INTEGRATION.md) for exact pinned revisions, all additions, benchmark denominators, validation evidence and limitations. Published benchmark results are owner artifacts, not independently reproduced frontend experiments. CSFCube and RECIPER have no sentence/verdict gold; they do not establish evidence F1 or verifier accuracy. Real-model attribution failed the numerical gate and remains flagged unstable.

## What was verified

**136 frontend tests, 163 current Yu Yue tests and pip check passed.** Actual recorded and fresh public searches each verified three papers / nine bounded semantic passages, with grounded explanations and replay. A constructed text PDF preserved exact page evidence through actual local NLI and replay. These are functional checks, not scientific accuracy studies. Historical 6 November QC remains in TEAM_QC_REPORT.md; its missing-selector finding is superseded.

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

Optional Qwen/Qwen2.5-0.5B-Instruct generation is local, Apache-2.0 and bounded by the same extractive gate. Run setup-local-llm.ps1 explicitly to install it on another machine. Its download is now complete and real local inference was tested. All 12 development/stress generations failed the strict schema and were safely replaced by templates; template explanations remain the default. Missing/unsupported local generation falls back visibly to templates. Scientific usefulness still requires human evaluation.

The evaluation harness, review rubric and existing synthetic cases remain available under `evaluation/`. Synthetic acceptance rates must not be reported as real LLM or scientific quality results. See [evaluation/rubric.md](evaluation/rubric.md).

## Configuration and handoffs

The built-in factory is `explanation_frontend.pipeline:create_provider`; no environment factory configuration is needed. Trusted server overrides EXPLANATION_PIPELINE_FACTORY and EXPLANATION_VERIFY_DISCOVERY_FACTORY remain supported. They are not upload-supplied code paths. The `.env.example` file documents settings and is not automatically loaded.

- [INTEGRATION_CHANGELOG.md](INTEGRATION_CHANGELOG.md): source revisions, complete additions log, policy changes and limitations.
- [RANKING_INTEGRATION.md](RANKING_INTEGRATION.md): manual ranking imports and contract.
- [PERSON4_INTEGRATION.md](PERSON4_INTEGRATION.md): native response and cached aggregator mapping.
- [contracts.md](contracts.md): local snapshots, citation lineage and factory interfaces.
- [API_POLICY.md](API_POLICY.md): no hosted/proprietary inference or paid API dependency.

Free anonymous public discovery APIs are the only claim-search network calls. Credentials are cleared. All embeddings, ranking, passage selection, verification, aggregation and explanations run locally. Provider access/rate-limit failures remain explicit; no paid fallback is used. Full-text reading, arbitrary PDF upload, independent scientific verification and production hosting are outside this implemented abstract pipeline.

## Team-wide QC and review packet

The latest [TEAM_QC_REPORT.md](TEAM_QC_REPORT.md) documents all five contributions: Raghavi 268/268; Naila 13/13 plus a confirmed cutoff-tie gap; Yu Yue 97/98 original and 98/98 with a scratch Windows-path fix; Chong Yao 5/5 actual-model controls; frontend 116/116. Suggested peer patches are in qc_suggestions and have not changed their branches.

Broader tests used 24 annotated SciFact-development pairs and 22 distinct claims against the full 5,183-document corpus. Retrieval was promising, but oracle rationale-label agreement was 12/24, the provisional selector's strict rationale recall was 16/27, and all 12 Qwen raw generations failed schema validation. All 24 development snapshot grounding gates and all 12 displayed generation fallbacks passed. These are development diagnostics, not untouched-test or human quality scores.

Open outputs/review_packet/index.html for the 24 paired raw outputs; upload review_packets.jsonl into Evaluate explanations, or use the blank review.csv. There are 48 pending rows for two reviewers. The private method key and generation records must remain hidden from independent reviewers until scoring. outputs/Kireeti_QC_Review_Packet.zip bundles the local materials. No human score was automatically assigned.

The public Kireeti_QC_Review_Packet.zip excludes private keys/statuses. Kireeti_QC_Coordinator_Packet.zip includes them for the study coordinator only.
