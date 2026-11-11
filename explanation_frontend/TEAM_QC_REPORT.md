Historical 6 November baseline. See [COMBINED_INTEGRATION.md](COMBINED_INTEGRATION.md) for the 9 November integration and current Yu Yue suite (163 passed). Missing-selector findings below are superseded; other unreviewed scientific limitations remain.

# Team-wide QC - 6 November 2026

The abstract-based app runs end to end. Retrieval is promising on the development sample, but passage selection, domain verification and optional explanation generation still need work. Software tests and mechanical grounding are not scientific quality scores.

All QC and fixes remain inside explanation_frontend on kireeti. Teammate branches were read, not changed. Peer patches were tested only in isolated copies.

## Task ownership and provenance

The [Project Role/Task Split](https://docs.google.com/document/d/1LoGlwoSwHwdG5S-0kq-lb8F2BmkJtAu0JpnAoHucpZY/edit?tab=t.hnn6tra4n34f) and [Edge Cases](https://docs.google.com/document/d/1LoGlwoSwHwdG5S-0kq-lb8F2BmkJtAu0JpnAoHucpZY/edit?tab=t.5yqorms9zhfn) were re-read on 6 November. **Person 3 owns evidence passage selection.** Person 4 explicitly consumes it. Yu Yue's README assigns it differently and should be corrected. RRF already exists in Naila's module; finishing passage selection is more urgent than duplicating fusion.

Revisions: Raghavi 259d2a9; Naila 2bc7d81; Yu Yue a91e257; Chong Yao 1ac06b6. Full results, packages and source hashes: [team_qc_summary.json](evaluation/team_qc_summary.json). Frozen cases and dataset hashes: [qc_development_manifest.json](evaluation/qc_development_manifest.json).

## Executed tests

| Owner | Original source result | Qualification |
| --- | --- | --- |
| Raghavi | **268/268 passed** | Module working directory; external network blocked, local Windows asyncio sockets allowed. |
| Naila | **13/13 passed** | Original tests use a hashing encoder. Additional real MiniLM evaluation below. |
| Yu Yue | **97/98 passed** | Windows provenance path-separator failure. Suggested patch gives **98/98** in a scratch copy. |
| Chong Yao | **5/5 passed**, actual model | Initialization adapted to pinned local CPU weights instead of downloading main. Original tests request threshold 0.60; production/development QC uses 0.50. |
| Kireeti | **116/116 passed** | UI, lineage, ranking order, relative manifests, pipeline, review-packet safety and existing regressions. pip check passes. |

Different suites are not interchangeable quality measurements. Initial Raghavi harness attempts used the wrong cwd or blocked Windows loopback; corrected runs passed. Those harness failures are not teammate defects. No human scores or approvals were invented.

## Person 1 - Raghavi

**Working:** broad tests, provider/query/date handling, deduplication, exact claim, canonical IDs, provenance, replay and failure diagnostics. The frontend's anonymous-provider path also completed a real fresh-search smoke.

**Missing:** pdf_verification.py explicitly declares PDF acquisition, extraction and evidence matching unimplemented. Full-text/page references, source eligibility/retractions and version/extraction-quality policy remain incomplete. The nested native records require an adapter to Naila's flat loader; the frontend supplies one. Optional upstream Gemini judging is proprietary and was not used in this app or QC.

Next steps:

1. **[PENDING - Raghavi]** Deliver safe PDF/text acquisition with page references and explicit unavailable/failed states if the project claims PDF support.
2. **[PENDING - Raghavi + Naila]** Agree the canonical-ID/nested-record contract rather than relying on field-name guesses.
3. **[PENDING - Raghavi + team]** Preserve source status/version/time, leave unknowns unknown, and manually review a separate live/PDF sample. Keep proprietary judging outside the free/open-source frontend.

## Person 2 - Naila

**Working:** shared BM25/SBERT/RRF interface, benchmark sentence lists, real full-corpus MiniLM indexing and frontend canonical-ID mapping. The development retrieval run did not filter candidates using gold labels.

**Confirmed defect:** _top_k partitions before sorting the selected subset. Scores [0,0,0,0,1,0,0,0,0,0], k=3 returned [4,0,2], despite documented index tie-breaking requiring [4,0,1]. Cutoff ties can change selected papers. The frontend requests complete branch lists and does not hit that cutoff; standalone top-K retrieval does.

Next steps:

1. **[PENDING - Naila]** Review [stable-cutoff patch](qc_suggestions/naila-stable-cutoff.patch). The original 13 tests plus an added cutoff regression pass **14/14** in an isolated patched copy. Published source is unchanged.
2. **[PENDING - Naila]** Freeze corpus/model/index/split settings and rerun the agreed evaluation after fixing the cutoff.
3. **[PENDING - Naila + Yu Yue]** Agree direct RankingInput exports and preserve branch ranks/scores. Inspect lexical/semantic failures; hybrid is not uniformly better than SBERT in the results below.

## Person 3 - Yu Yue

**Working:** deterministic RRF, identity checks, scorer boundary, evaluation/timing tools. Both owners' RRF produced identical top-100 IDs on **22/22** real fixed-corpus inputs. The optional scorer returned finite float32 and float64 scores here; the prior NaN issue was not reproduced in this environment.

**Confirmed defect:** benchmark provenance uses str(relative_path), yielding Windows backslashes while the test expects POSIX keys. The [as_posix patch](qc_suggestions/yuyue-windows-provenance.patch) gives 98/98 in an isolated copy.

**Main missing responsibility:** the published module has no final source-linked passage selector, baseline/context comparison or annotated passage precision/recall/F1. Kireeti's temporary selector keeps the app runnable but does not complete Person 3's assignment.

Next steps:

1. **[PENDING - Yu Yue]** Apply/review the portability patch and correct passage ownership in the README.
2. **[PENDING - Yu Yue]** Implement/approve sentence baseline and contextual selection with stable source references. Evaluate alternative rationales, context/attribution, truncation and relevant-paper/irrelevant-passage failures.
3. **[PENDING - Yu Yue]** Reproduce the pinned scorer and compare rerank on/off on the agreed split. Small development results improved here; production remains RRF-only pending broader review.

## Person 4 - Chong Yao

**Working:** real support/contradiction controls, input orientation, label mapping, truncation flags/exclusion, per-paper conflicts and cached aggregation. All five model-backed tests pass.

**Standalone contract gaps:** verify accepts a passage with a paper_id different from its enclosing selected paper. Schemas do not enforce all identity/probability constraints. The frontend rejects that lineage mismatch. Original aggregate ignores internal-paper conflict at corpus level and can drop all contradiction evidence when slicing five primary passages. The frontend's explicit safe_aggregate policy preserves both sides. Overlap/source-study deduplication is not established by the vote cap.

**Quality concern:** only **12/24 annotated development rationale labels matched**. The other predictions were NEUTRAL: 7 support cases and 5 contradiction cases. Zero opposite-sign errors in this tiny sample is not a scientific safety guarantee. The original calibration script creates extra neutral examples from the first two sentences of cited non-evidence documents; these are heuristic neutral examples, not independently annotated sentence-level NEUTRAL labels. Report that construction when presenting its scores.

Next steps:

1. **[PENDING - Chong Yao]** Validate IDs and finite probabilities before inference; test duplicates/overlap, source versions and mixed failures.
2. **[PENDING - Chong Yao]** Review the conflict-side policy and expose native operational status separately from scientific insufficiency.
3. **[PENDING - Chong Yao]** Investigate domain abstention and neutral-data validity. Compare annotated evidence with genuinely retrieved evidence, report class confusion/failure rates and freeze thresholds before final evaluation.

## Person 5 - Kireeti

**Working:** integrated abstract pipeline, UI, exact citations/quotation gate, conflicts/failures, cached subsets, templates, replay and human-review collection. All 24 development snapshots passed mechanical grounding/export checks.

**Fixed in QC:** verification paper order follows RRF; relative replay paths resolve against their manifest; an unknown paper version is no longer filled with a discovery/code reference; unknown retrieval time is labelled honestly; large ranking diagnostics are collapsed. Early real-result exports now replay with only the known missing safety note restored; arbitrary limitation changes still fail. The UI declares the English-model scope. Generation settings record CPU threads/tokens without stale counts. Review HTML escapes source text and scoring rows remain blank.

**LLM outcome:** Qwen installation and actual local inference now work, but **0/12 raw outputs passed the strict schema** (6 synthetic stress cases, 6 real development contexts). **12/12 displayed fallbacks passed** the grounding gate. This validates safe rejection, not good LLM explanations. Templates stay default; the optional control is labelled experimental.

Next steps:

1. **[PENDING - Kireeti + reviewers]** Score the prepared 24 paired raw outputs and separately inspect displayed fallbacks. There are **48 blank human-review rows**, zero completed scores.
2. **[PENDING - Kireeti]** Improve schema-constrained generation on development cases, then evaluate untouched cases. Do not claim an LLM advantage from template fallbacks.
3. **[PENDING - Kireeti + team]** Review shared policies, accessibility/manual usability and final real-claim acceptance cases.

## Frozen development measurements

The manifest was saved before inference: SHA-256 ordering, 12 annotated SUPPORT and 12 CONTRADICT rationales, **24 pairs / 22 distinct SciFact-dev claims** and the full **5,183-document** retrieval corpus. Person 4 already calibrated on dev, so these are **development diagnostics, not untouched test results**. Threshold stayed 0.50; no tuning or heuristic neutral labels were added.

| Method | MRR@100 | Recall@10 | Recall@100 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.7218 | 0.7250 | 0.9682 | 0.6933 |
| SBERT | 0.7806 | 0.9273 | 1.0000 | 0.8124 |
| RRF | 0.8062 | 0.9273 | 1.0000 | 0.7890 |
| RRF + cross-encoder top 20 | 0.9106 | 0.9500 | unchanged top-100 scope | 0.9026 |

Indexing took about 100.51 s. Mean warm query scoring was 0.28 ms BM25, 6.66 ms SBERT and 6.54 ms RRF; optional reranking averaged 0.607 s/claim. These exclude model/index construction and are not end-to-end percentiles. SBERT exceeded RRF on nDCG in this sample, while RRF improved MRR.

The provisional selector fully covered 106 original sentences against 27 chosen-rationale sentences, with 16 matches: micro precision **0.1509**, recall **0.5926**, F1 **0.2406**. Source spans matched in 24/24 cases; none selected a partial original sentence. This strict custom diagnostic penalises context neighbours and does not credit every alternative rationale. It is **not official SciFact evidence F1**, but shows substantial missed rationale coverage.

Selected-window paper verdicts agreed with the chosen rationale label in 18/24 cases (75%). Gold papers were supplied and context differed from the oracle-rationale setting: this is not full-pipeline accuracy or a true oracle-to-pipeline gap. Mechanical grounding passed 24/24, which does not establish relevance or scientific support. Cached aggregation averaged about 0.081 ms across 100 repeats per small one-paper snapshot; the earlier full 3-paper fresh-search smoke was about 10.43 ms. Keep timing scope explicit.

## Shared review order

1. **[PENDING - team]** Agree one split protocol. Naila proposes train tuning/dev reporting; Chong Yao calibrated on dev. Treat current dev measurements as development. Agree an untouched labelled holdout or separately reviewed live sample before final claims; official hidden test labels are unavailable locally.
2. **[PENDING - Yu Yue]** Finish evidence selection, the clearest missing stage.
3. **[PENDING - Chong Yao]** Address abstention, neutral labels and native aggregation/input contracts.
4. **[PENDING - Naila / Yu Yue]** Review the two proposed patches and publish direct shared handoffs. Scratch fixes have not changed their branches.
5. **[PENDING - Raghavi]** Complete text acquisition if claiming PDFs; otherwise explicitly scope the demo as abstract/description-only.
6. **[PENDING - Kireeti + reviewers]** Complete human reviews, retaining unsuccessful generations in the results.

## Reproduction and artifacts

After setup-pipeline.ps1 and git fetch origin, run with the module Python:

```powershell
python -m explanation_frontend.evaluation.team_qc raghavi
python -m explanation_frontend.evaluation.team_qc naila
python -m explanation_frontend.evaluation.team_qc yuyue
python -m explanation_frontend.evaluation.team_qc chong_yao
python -m explanation_frontend.evaluation.team_qc yuyue --portability-fix
python -m explanation_frontend.evaluation.team_qc naila --stable-cutoff-fix
python -m explanation_frontend.evaluation.development_qc
python -m explanation_frontend.evaluation.selection_qc
python -m explanation_frontend.evaluation.contract_probes
python -m explanation_frontend.evaluation.reranker_qc
```

Use explanation_frontend/.venv/Scripts/python.exe, not an unrelated global environment. reranker_qc explicitly downloads a pinned Apache-2.0 model if absent, verifies its hash and loads locally; it does not enable product reranking. Raw logs/JUnit XML, metric records and generation records are under ignored outputs/team_qc, outputs/development_qc and outputs/qwen_qc.

Review materials: outputs/review_packet/index.html and outputs/Kireeti_QC_Review_Packet.zip. The private subdirectory contains method keys/statuses; keep it away from independent reviewers until scoring ends. No human scores were filled, no external messages sent and no teammate/main branch changed.

The public Kireeti_QC_Review_Packet.zip excludes private keys/statuses. Kireeti_QC_Coordinator_Packet.zip includes them for the study coordinator only.
