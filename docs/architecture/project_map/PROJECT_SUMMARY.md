# Scientific claim verification: the integrated project

**Snapshot: `abaaff7`, 11 November 2026.** [Open the interactive architecture map](project-summary.html).

Imagine a research assistant who keeps the books open while answering. You give it one scientific statement. It finds possible papers, picks passages to inspect, asks a local model whether those passages support or contradict the statement, and places the answer beside its exact sources. You can change which papers count and save the whole result for replay.

That is this project's purpose: **help people inspect scientific claims against traceable evidence**. Its strongest contribution is the combined workflow and its visibility into sources, decisions and limitations. It is a research prototype with measured performance and remaining errors. The displayed verdict is a model's interpretation of the selected evidence, not a guarantee of scientific truth.

## The user journey

1. Start in **Project overview**, or open **Find papers**. Enter a claim for fresh search, load a recorded real search, or import a saved result. The original claim stays authoritative even when search queries are reformulated.
2. Inspect candidate papers, provider provenance and available abstracts. New advisory checks highlight missing, possibly cut-off, body-like or uncertain descriptions. Users decide what to include.
3. Prepare the selected papers and open **Verify evidence**. Local retrieval ranks them, evidence selection prepares exact source passages, and DeBERTa predicts passage relationships.
4. Inspect the provisional all-paper result, individual paper decisions, exact passages and the verdict over the current paper selection. Supporting and contradicting evidence can remain visible together.
5. Change the selection to compare evidence scope. Stored predictions are reused. Export the result as JSON, then reload it through **Cached replay**.
6. Use the separate **Local PDF**, **Model attribution**, **Benchmark results** and **Evaluate explanations** workspaces when needed. The app has seven workspaces in total. [Application routing][app]

## How the five contributions fit together

| Person | Implemented contribution | Open scientific or review work |
|---|---|---|
| **Raghavi** | Public catalogue discovery, query generation, normalized metadata, canonical identities, deduplication and provider provenance. Her latest abstract-quality rules are integrated as an advisory sidecar. | Measure live-provider recall on shared queries and validate warning rules against human labels. Provider availability and description quality remain variable. |
| **Naila** | Custom NumPy BM25 and MiniLM/SBERT retrieval. Experimental Captum IG is connected; operational word-MASK sensitivity is available through the integrated frontend. | Investigate retrieval misses and broader XAI plausibility/faithfulness. IG numerical reliability remains unresolved for the real cases tested. |
| **Yu Yue** | RRF fusion, frozen semantic sentence selection, exact spans, verifier budgets, PDF evidence support and three-domain retrieval evaluation. The bounded-context candidate is the current primary abstract policy. | Improve rationale coverage and contradiction-recall tradeoffs using separate development data. Review PDF reading order and sentence splitting. |
| **Chong Yao** | Local DeBERTa NLI, passage/paper decisions and decoupled verification/aggregation APIs. Cached native responses support subset changes. | Investigate supported/neutral and contradiction errors, population shifts and confidence calibration. |
| **Kireeti** | Streamlit integration, typed contracts, grounded explanation templates, citation checks, scope controls, background jobs, PDF/replay bridges, quality-warning presentation and evaluation/review exports. | Complete human explanation/source/PDF reviews, review the error cases, and rehearse a truthful end-to-end demonstration. |

These are implementation and evidence-based statuses, not invented completion percentages. Upstream modules are pinned inside the owned frontend folder or installed at a fixed revision. [Integration record][integration] [Pinned sources][pins]

## The active technical pipeline

### 1. Discover papers while preserving the claim

Raghavi's pinned `paper_search` package uses local query-generation models and public OpenAlex, OpenAIRE and arXiv catalogues. The frontend clears credentials, disables other providers and keeps the optional arXiv fallback disabled. Records retain original claim text, identities, aliases, source records, query ranks and operational diagnostics.

**Fresh discovery is an external network operation; model inference is local.** A paper URL or PDF metadata link does not mean the full paper was read. Find papers allows up to 30 candidates per provider; Verify evidence's direct Live route requests ten per provider. [Discovery factory][discovery] [Live pipeline][pipeline]

### 2. Combine lexical and semantic relevance

BM25 rewards informative keyword overlap while accounting for term frequency and document length. Its implementation is custom NumPy code, with production Porter stemming. SBERT represents the claim and papers with MiniLM embeddings; cosine similarity estimates semantic relevance.

Yu Yue's Reciprocal Rank Fusion combines their ordered results:

\[
\operatorname{RRF}(d)=\sum_{m\in\{BM25,SBERT\}}\frac{1}{60+\operatorname{rank}_m(d)}.
\]

This combines positions without pretending the two raw score scales are equivalent. Uploaded rankings must match the exact loaded search and reproduce the pinned fusion. Cross-encoder reranking is not enabled in the active frontend.

The interactive app ranks the selected available records. **It does not universally choose five papers.** Top-100 branches followed by automatic top-five verification are settings of the offline evaluation wrapper. [Retrieval bridge][retrieval] [Ranking validation][ranking]

### 3. Select exact evidence within a real token budget

The current app factory is `primary_candidate:create_provider`. The original `LocalPipeline` baseline remains available through an explicit override. The primary abstract route performs:

1. Semantic sentence-anchor selection: top three, cosine threshold 0.5, context radius zero.
2. Actual tokenization of the whole original abstract with the original claim.
3. Exact whole-abstract context when the pair fits 512 tokens, including special tokens; otherwise the original focused passages.
4. A relevance gate requiring a maximum budget-eligible proposed span score of at least 0.65. Otherwise the selector abstains.
5. Final budget checks that reject whole oversized passages instead of silently trimming their source text.

The whole-abstract score inherits the best anchor score. The 0.65 gate is a relevance condition, not DeBERTa confidence or a truth probability. Passage identifiers encode source offsets and text hashes. [Primary policy][primary] [Context selection][context] [Budget contract][budget]

### 4. Verify passage relationships locally

DeBERTa receives **evidence as premise** and **the original claim as hypothesis**. Its three passage labels are `SUPPORT`, `CONTRADICT` and `NEUTRAL`. Support/contradiction predictions below the frozen 0.50 confidence threshold downgrade to neutral. The selected model is loaded locally in evaluation mode using CPU float32 and a pinned checkpoint revision.

The integrated selector preflights input length. The owner's raw classifier also detects and flags truncation if called with oversized input outside that preflight. Operational failures are retained separately from neutral or insufficient evidence. [Local model loading][model] [Classifier][classifier]

### 5. Aggregate, explain and preserve the result

Passage decisions aggregate within papers, then across the selected paper scope. The final vocabulary is **Supported, Contradicted, Conflicting or Insufficient**. Frontend conflict safeguards preserve evidence from both sides rather than losing one side under an evidence-count cap.

The default explanation is a deterministic template containing whole exact primary-evidence passages, citation IDs and scope limitations. Validation checks the current context, verdict, citations and exact passage equality. It requires both support and contradiction in a conflicting explanation. This establishes mechanical grounding and consistency, not independent semantic correctness. [Aggregation adapter][adapter] [Templates][templates] [Grounding gate][grounding]

## Other connected routes

**Uploaded PDFs.** PyMuPDF extracts text from an explicitly uploaded local PDF. Semantic selection and the same verifier input discipline retain page references and source hashes. Limits are 20 MiB, 200 pages and two million extracted characters. Scanned documents, OCR and image/table interpretation are unsupported. PDF selection is separate from the promoted whole-abstract policy. [PDF route][pdf]

**Replay and caching.** Live/selected-search sessions cache the original native verification response, then validate the snapshot and paper IDs before reaggregating it. A JSON replay is a different route: its unchanged selection restores saved aggregation; changed selections use the explicitly labelled local replay policy. Export preserves the snapshot, selected IDs, aggregation, explanation and checksum. Checksums detect changes, not source authenticity. [Cached adapter][adapter] [Replay policy][replay] [Export/import][export]

**Abstract quality.** The new offline rules produce advisory flags and a separate JSON report without changing text, selection, ranking, model input or predictions. Candidate counts and provider occurrences remain separate. Missing legacy provider records are disclosed, and orphan occurrences can withhold only that report scope. The owned batch runner fixes empty-result CSV reporting. “No warning detected” is deliberately weaker than “valid abstract.” [Quality integration][quality]

**Word sensitivity and IG.** Default word-MASK occlusion measures target-probability change when one whole word is replaced. It scores up to 96 words, balances claim/evidence coverage and reports exact offsets. Its default target is the raw highest-scoring class before the confidence threshold, which can differ from a thresholded Neutral decision. Scores need not add up to the prediction and do not prove causation. Captum IG remains experimental, with strict finite, truncation and completeness gates; unstable scores are withheld. [Occlusion][occlusion] [IG gates][ig]

**Optional local generation.** Qwen2.5-0.5B-Instruct can attempt a structured explanation locally. Its tested raw outputs were rejected in all 12 QC cases. The template fallback remains available; fallback success is not evidence of good LLM generation. [Generation and validation][llm] [QC record][qc]

## Dependencies, contracts and deployment boundaries

| Layer | Main dependencies or model |
|---|---|
| UI and contracts | Python setup recommends 3.12; Streamlit 1.58.0; Pydantic 2.12.5 |
| Discovery | `paper_search`, spaCy, KeyBERT, httpx; public catalogue services |
| Retrieval/selection | NumPy, NLTK Porter, Sentence-Transformers, `all-MiniLM-L6-v2` |
| Verification | PyTorch, Transformers, `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| PDF and optional XAI | PyMuPDF; Captum for experimental IG |
| Optional generation/testing | Local Qwen2.5-0.5B-Instruct; pytest 9.0.2 |

NLI is pinned to `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`. The production retrieval encoder uses MiniLM revision `2a7cca6ad7605f3183bce07f677bda658059bbe9`, while the semantic selector uses revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Discovery stays at Raghavi `259d2a9`, its independent quality sidecar at `0ee9565`, Naila retrieval at `2bc7d8`, Yu Yue's base modules at `7ed93fa`, bounded context at `ba8cdd6`, and CY verification at `1ac06b6`. Source records document packaging and integration adjustments. [Source manifest index][pins] [Retrieval model pin][retrieval-model]

Runtime persistence is local JSON, model/artifact files, Streamlit session state and in-memory caches. There is no production database or distributed service cluster to imply. Setup explicitly downloads public models; missing models are not silently downloaded during verification. Hosted inference and paid API dependencies are absent from the active workflow. [API policy][api]

Pydantic contracts retain paper/passage links and finite probability bounds. Context keys bind the snapshot, selection and aggregation. Background request tokens prevent stale results from attaching to a changed claim. Deadlines and cancellation discard late results, while abandoned threads may finish in the background. Model loading is cached and neural operations are locked. [Contracts][contracts] [Job state][state]

## What the measurements establish

### Current primary policy: 209-record internal audit

The frozen comparison uses 209 reserved claims from published SciFact training data, with cited/evidence-paper IDs disjoint from logged prior training and inspected development work. It uses the full 5,183-paper corpus and unchanged retrieval/NLI settings. This is **an internal audit, not an official hidden test**.

| Metric | Original baseline | Current bounded-context candidate |
|---|---:|---:|
| Accuracy | 56.5% | 67.9% |
| Macro F1 | 0.583 | 0.679 |
| Contradiction precision | 47.8% | 84.0% |
| Contradiction recall | 82.1% | 53.8% |
| Input evidence F1 | 0.210 | 0.165 |

The candidate fixes 45 errors and introduces 21 regressions, with zero processing failures. The paired accuracy gain is 11.5 percentage points; its cited-paper-cluster 95% interval is +3.8 to +18.7 points. Better overall accuracy therefore comes with worse contradiction recall and evidence F1. Pretrained exposure and prior inspection of published records cannot be ruled out. [Frozen audit][audit]

### Three-domain retrieval and separate research

| Dataset | RRF Recall@10 | MRR@10 | nDCG@10 | Evaluation scope |
|---|---:|---:|---:|---|
| SciFact dev | 95.6% | 0.806 | 0.833 | 188 positive claims, full 5,183-paper corpus |
| CSFCube fold1 test | 39.2% | 0.648 | 0.438 | 26 query-facet tasks, judged candidate pools |
| RECIPER | 93.7% | 0.818 | 0.847 | 1,024 questions over 343 paper titles/abstracts |

Recall measures target coverage, MRR rewards finding a relevant paper early, and nDCG rewards placing relevant papers high in the list. Protocols and encoder/preprocessing settings differ, so these are separate dataset results. Only SciFact supplies the verdict/rationale labels used here. These numbers do not demonstrate three-domain verification accuracy or live catalogue recall. [Full validation report][validation]

The earlier 300-claim SciFact dev baseline scored 56.7% accuracy. A separate exploratory guarded-context study reached 66.0% with 79.7% contradiction recall, preserving some decisive baseline decisions. It was not promoted merely for that development-set tradeoff. These 300-claim results must not be mixed with the 209-record audit. The four-pair occlusion pilot, drawn from three documents, is likewise exploratory: sentence AP 0.833 versus random 0.438 is not broad XAI validation. [Context research][research] [Occlusion pilot][pilot]

### Software reliability is a different result

The current integration passes **212 frontend tests** and dependency validation, including 31 new quality/provenance cases. The Desktop checkout separately passed those 31 cases. Historical isolated owner suites passed 452 tests across four owners; they are separate runs, not extra current scientific examples. Earlier bounded checks include seven PDF cases, 30 repeated retrieval lists and five repeated NLI pairs. Synthetic scientific-language expectations passed 12 of 15 cases, with population/generalisation errors retained. Current-primary instrumentation recorded 20 subset changes with zero additional neural forwards. None of these proves scientific accuracy. [Current integration validation][integration] [Scientific/reliability report][validation]

## Remaining work before recording

The end-to-end software is integrated. Remaining work is concentrated in scientific review, communication and targeted improvements:

1. Review representative successes and failures, including all 21 audit regressions. A nine-case review pack exists, but **zero ratings are completed in the prepared explanation-review study**. Record genuine usefulness/citation judgments and inspect PDF page correctness.
2. Rehearse the current-primary app route and prepare a labelled replay fallback. The curated overview candidate pool contains an annotated source plus seeded distractors; disclose it as an integration demonstration rather than retrieval evaluation.
3. Keep the measured cohort, class tradeoffs and source scope visible. A high confidence score can still accompany a wrong population/generalisation judgment.
4. Freeze the demonstrated version and inputs, and check any local factory overrides. The executable default is the promoted primary provider; the documentation-only `.env.example` still names the retained baseline in an old comment. Further contradiction-recall, calibration or selector changes need separate development data and a new frozen comparison, not tuning on the inspected audit. [Configuration comment][config-comment]
5. Validate broader XAI and abstract-quality performance with human labels. Keep experimental IG and Qwen claims proportional to the negative/limited evidence.

Live-provider recall, non-English performance, source authenticity, retractions, PDF reading order and human explanation quality remain limits. The immediate deliverable is a reproducible, inspectable evidence workflow with measured gains and explicit tradeoffs. [Regression review status][review]

[app]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/app.py#L23
[integration]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/FINAL_INTEGRATION.md#L1
[pins]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/_upstream/README.md#L1
[discovery]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/discovery.py#L227
[pipeline]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/pipeline.py#L91
[retrieval]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/pipeline.py#L117
[ranking]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/ranking.py#L26
[primary]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/primary_candidate.py#L9
[context]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/_upstream/person3/contextual_evidence.py#L47
[budget]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/_upstream/person3/verifier_budget.py#L26
[model]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/pipeline.py#L27
[classifier]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/_upstream/claim_verifier/verifier.py#L80
[adapter]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/person4.py#L83
[templates]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/templates.py#L8
[grounding]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/validation.py#L17
[pdf]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/pdf_view.py#L21
[replay]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/providers.py#L22
[export]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/export.py#L20
[quality]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/ABSTRACT_QUALITY.md#L1
[occlusion]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/occlusion.py#L5
[ig]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/xai_runtime.py#L11
[llm]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/llm.py#L117
[qc]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/TEAM_QC_REPORT.md#L1
[api]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/API_POLICY.md#L1
[contracts]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/models.py#L107
[state]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/state.py#L15
[audit]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/evaluation/independent_results_20261111/REPORT.md#L1
[validation]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/evaluation/validation/FINAL_TEST_REPORT.md#L1
[research]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/evaluation/context_study/REPORT.md#L1
[pilot]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/evaluation/occlusion-results.json#L1
[review]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/evaluation/regression_review_20261111/SUMMARY.json#L1
[retrieval-model]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/INTEGRATION_CHANGELOG.md#L51
[config-comment]: https://github.com/YYU035/NLP_Project/blob/abaaff7cd76749a8d93e75c7d747cb349e1f5f91/explanation_frontend/.env.example#L13
