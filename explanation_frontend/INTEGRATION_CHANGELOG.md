# End-to-end integration additions - 6 November 2026

Branch: **kireeti**. All source additions are under explanation_frontend. No teammate source directory, main branch, model weights or datasets are merged into Git. Pinned runtime copies are used because the team modules live on independent branches and this user's scope is restricted to this folder.

## Pinned teammate sources

| Owner | Upstream revision | Integration |
| --- | --- | --- |
| Raghavi | 259d2a9e90fb8e9d5988f030a156733f4f3576d1 | Existing optional paper_search dependency, live discovery bridge and recorded search. |
| Naila | 2bc7d81292313498fd70dd13656024d00cf3dda4 | Six unchanged runtime files in _upstream/retrieval: __init__, schema, bm25, sbert, retriever, data. No benchmark data or downloads copied. |
| Yu Yue | a91e257ab260ded8df8c2cc2f214c4c7fd16af6c | Existing six unchanged ranking runtime files in _upstream/ranking_evaluation. Actual Naila rankings now feed RankingInput. |
| Chong Yao | 1ac06b675e3d0719a63cb8efe98eda4f2c5fd576 | Existing _upstream/claim_verifier schemas and runtime; only packaging change is relative .schemas import. Classification/verify methods remain the owner's implementation. |

## New files and changes

| Addition | Purpose and design decision |
| --- | --- |
| pipeline.py | Built-in factory for both Live and Selected search. Converts nested discovery records directly to Naila Paper while preserving canonical identity and the entire original discovery payload. Passes exact claim unchanged through all stages. |
| Local encoder construction | Uses the existing pinned local MiniLM files with local_files_only=True and trust_remote_code=False. No implicit network or embedding cache outside this module. Naila's optional constructor download path is bypassed by injecting this encoder. |
| BM25/SBERT to RRF adapter | Builds both indexes on selected candidates with text, takes complete per-branch ranked IDs, converts query ms to seconds and invokes Yu Yue's actual RRF. Existing validated supplied rankings can be reused. Original metadata, branch scores, ranked output and source commits are retained in the result. |
| passage_selection.py | Missing-stage fallback, added by Kireeti: deterministic, BM25-selected adjacent-sentence windows from provider abstract/description text. At most three per paper. Text is never rewritten; passage IDs record exact original offsets and text digest. Long sentences remain whole and are excluded by the owner's NLI truncation guard when necessary. This is not the team's final scientific passage selector. |
| Local NLI construction | Uses Chong Yao's classifier/verify methods with explicit offline tokenizer/model objects. CPU float32, max 512 tokens, threshold 0.50. Model metadata records immutable revision; no hosted inference or runtime download. The shared model forward pass is locked; each provider caches its own typed response. |
| safe_aggregate | Calls the owner's cached aggregator. Adds an explicit frontend policy for internal-paper conflict and for the five-evidence cap: retain at least one eligible SUPPORT and CONTRADICT span, then fill remaining slots up to five. Missing conflict-side evidence fails closed. This changes the owner's aggregation behavior only for conflict cases and requires team acceptance; it is recorded in verdict_delta_reason. |
| Person4Adapter update | Allows an explicit CONFLICTING aggregation for internally conflicted papers; existing primary-evidence validation still requires both sides. Other unresolved/conflicting outputs remain rejected. |
| Operational and empty-input handling | Papers without usable passages remain visible with safe structured failures. No usable selected evidence or total discovery failure gives failed processing with no scientific verdict. Successful empty discovery yields NO_RETRIEVAL; empty paper selection after a populated run yields EMPTY_SELECTION. Search partiality remains visible separately. |
| pipeline_setup.py / setup-pipeline.ps1 | Explicit public model setup. Downloads the reviewed MIT NLI files at pinned revision and checks the weight SHA-256 before writing the activation marker. Existing discovery setup supplies runtime dependencies and embeddings. No new package downgrade or hosted API dependency. |
| pipeline_smoke.py | Real-model recorded and fresh-search checks, scoped subsets, grounding gate and JSON replay round-trip. Outputs are ignored local artifacts. |
| verification_view.py | Built-in factory is now the default. Live and Selected search work without hand-written environment factory setup. Initial pipeline deadline is 300 seconds for CPU/cold-load work. Existing cancellation, stale-result checks and five-second aggregation deadline remain. |
| templates.py | Automatically discloses that selected spans come from unverified provider abstracts/descriptions and that full text/source authenticity were not checked. |
| run.ps1 | Enables UTF-8 Python output for claims containing scientific symbols. |
| tests/test_pipeline.py | Whole-chain contracts, exact offset/negation preservation, no-text inputs, total provider failure, internal conflict, no repeated classification on subset changes, replay round-trip and Streamlit default-factory workflow. Model/encoder doubles are clearly synthetic. |
| Documentation | README checklist refreshed; this ledger, policy and handoff guides distinguish actual integrated behavior from optional/future work. |

## Model provenance

- Embeddings: sentence-transformers/all-MiniLM-L6-v2, revision 2a7cca6ad7605f3183bce07f677bda658059bbe9, existing local installation.
- NLI: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, revision b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7. The pinned model card declares MIT: https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli/blob/b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7/README.md
- NLI model.safetensors: 870182606 bytes, SHA-256 c03cd208bf920b4fbbb182a0535859a566e8acc3477d2b536bf87b769978524b. Weights are ignored under .models/claim-verifier.
- Optional Qwen explanation model is unchanged and not needed for the working template pipeline. Its earlier incomplete download is not claimed as a successful LLM experiment.

## Deliberate limits and team follow-up

- This is an abstract/description-only pipeline. No arbitrary PDF is downloaded, no full-text passage is fabricated, and source authenticity/retraction/freshness is not established.
- RRF scores, retrieval scores and NLI confidence are not probabilities of scientific truth. Exact quotation/citation validation is not semantic or scientific evaluation.
- Naila's benchmark output schema and Raghavi's nested records differ; the frontend maps through canonical identity and keeps original records rather than assuming they are interchangeable.
- The fallback selector needs approval/replacement by the passage owner. It does not claim SciFact rationale sentence indices or evidence-F1 performance.
- The added conflict policy must be reviewed with Chong Yao. Original typed per-paper/passages remain cached for traceability; no extra inference occurs on paper-subset changes.
- Two real smoke claims establish functioning wiring, not calibrated scientific accuracy. Human/held-out evaluation, full-text work and optional reranker/Qwen comparisons remain pending.
- Changed offline replay selections still use the documented local replay policy; the exact owner's native cache is session-owned and is not reconstructed from an export.

## Team-wide QC follow-up - 6 November 2026

- Added isolated upstream test runners (evaluation/team_qc.py), independent contract probes, frozen development-corpus checks, strict source-sentence coverage diagnostics, real optional reranker QC and offline human review-bundle generation.
- Completed pinned public Qwen download and real offline inference: all 12 raw outputs failed the schema, all displayed fallbacks passed. No automatic repair or fabricated quality score was added; the UI now labels this optional path experimental.
- Fixed frontend paper display order to match ranking, relative replay paths to use their manifest directory, unknown paper-version provenance, and unknown retrieval-time labelling. Moved bulky ranking JSON into a collapsed diagnostic. Added non-stale generation token/thread settings.
- Re-read the task/edge-case document; Person 3 explicitly owns passage selection. The fallback remains provisional. TEAM_QC_REPORT.md records that gap, real metrics, failures and suggested next steps for every owner.
- Created proposed Naila cutoff-tie and Yu Yue Windows-provenance patches under qc_suggestions. Only scratch test copies were patched; teammate source branches and the product's pinned copies are unchanged.
- Prepared 24 paired raw-output packets and 48 blank human-review rows, with a private method key, escaped offline HTML and a local ZIP. Raw/generated/review artifacts and model weights remain ignored under this folder.
- Machine-readable summary and the pre-inference frozen development manifest are versioned under evaluation. Current frontend suite: 116 passing tests; dependency check passes. Dataset is development only and already used by an owner for calibration; no claim of independent final-test performance or completed human review.

- Additional compatibility fix: known early integrated replays regain only the exact missing source-authenticity safety note. Both saved real smoke exports now validate; arbitrary limitation tampering remains rejected. Added a regression and an explicit English-model scope caption.
