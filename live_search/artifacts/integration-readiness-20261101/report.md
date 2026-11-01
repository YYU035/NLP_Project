# Person 1 retrieval readiness — 1 November 2026

Implemented the requested provider policy in an **additive handoff service**: OpenAlex → KeyBERT; OpenAIRE → KeyBERT; arXiv → spaCy. Existing `SearchPipeline`, query-generator modes, client defaults, retries, pacing, and all prior experiment/label files remain unchanged. **No live requests or new relevance judgments were made.**

## 1. Baseline audit: completed comparisons

Counts below are retrieved claim–paper pairs, deduplicated within each claim, except the 30 October report's provider-wide distinct-paper totals. Provider rows are not globally additive. These are separate runs, not independent replications of one pooled trial.

| Saved run | Methods / budget | OpenAlex | arXiv | OpenAIRE | Interpretation |
|---|---|---|---|---|---|
| [30 Oct query evaluation](../query-evaluation-20261030/report.md) | Full claim vs earlier deterministic reporting-frame alternative; top 5; alternatives for 7/10 claims; repeats also present | Original 50 → query union 63 | Original 5 → union 5 | Not tested | Earlier query construction, not the later two-concept spaCy baseline. 72 logical / 116 HTTP calls across providers/arms/repeats; Semantic Scholar rate limits prevent a clean three-provider conclusion. |
| [31 Oct concept rendering](../arxiv-concepts-20261031/report.md) | Original vs spaCy; 10 claims; top 10; 10 calls/method/provider, 40 total | 100 vs 100; overlap 9; union 191 | 10 vs 40; overlap 5; union 45 | Not tested | arXiv original arm is legacy every-token AND, **not** current concept-rendered original input. Overall 250 occurrences / 234 pairs. |
| [1 Nov OpenAIRE](../openaire-experiment-20261101/report.md) | Original vs spaCy; same claims/dates/top 10; 20 calls | Saved comparator only | Saved comparator only | 14 vs 100; union 112; 94 historically new | Seven original zeros, no concept zeros; 94 novel assessments still blank. |
| [1 Nov matched KeyBERT](../keybert-experiment-20261101/final_strategy_comparison.md) | Fresh spaCy vs KeyBERT; same claims/dates/top 10; 40 calls | 100 vs 100; overlap 22; union 178 | Not tested in this run | 100 vs 81; overlap 13; union 168 | Overall 381 occurrences / 324 pairs; all calls succeeded. `kind=original` here means **spaCy**, not full claim. |
| [1 Nov arXiv KeyBERT](../arxiv-keybert-20261101/report.md) | Fresh spaCy vs KeyBERT; existing explicit phrase/literal rendering; 20 calls | Not tested | 40 vs 12; overlap 10; union 42; zero queries 6 vs 8 | Not tested | All calls succeeded; all 42 have abstracts and PDF URLs. Union adds two unassessed pairs but no additional known relevant pairs. |

The matched OpenAlex comparison has known provisional relevant/evidence yields **90/47 spaCy vs 93/67 KeyBERT**, with 6 vs 3 unresolved. OpenAIRE has **17/10 vs 64/39**, but **83 vs 10 unresolved**: its relevance comparison remains incomplete. arXiv has **29/21 vs 9/7**, with 5 vs 3 unresolved. These are provisional AI-assisted judgments, not established accuracy or verified evidence. The requested policy is reasonable to integrate as a configurable choice; OpenAIRE superiority is not established.

Missing comparisons: a same-run full-claim/spaCy/KeyBERT three-way comparison under current rendering; evaluated arXiv fallback; top-20/top-30 yields; validated labels; query-model CPU/RAM latency; end-to-end Person 2 ranking. Pairwise comparisons already exist for all requested providers: **do not rerun them merely to fill a table**. Earlier diagnostic/smoke requests and query-only drafts are not additional strategy trials. No reliable relevance evaluation was performed by the Gemini API (saved judgments are blank).

## 2. Completed code changes

- [`retrieval.py`](../../src/paper_search/retrieval.py): `RetrievalService.retrieve(claim, top_k=30)` returns Pydantic normalized candidates, unchanged original claim, identity aliases, merged metadata, all normalized source records, provider/query/rank provenance, exact safe HTTP parameters, attempts, errors and elapsed time. Cross-provider dedup reuses the existing conservative matcher. **top_k is per-provider depth**, giving up to 90 unranked candidates across three providers, not a ranked global top 30.
- Existing clients retain timeout/retry/rate-limit handling. Providers fail independently; valid zeros remain distinct from failures. Complete failure raises `RetrievalUnavailable` with its diagnostic `result`. Missing local query models fail explicitly per provider; there is no silent strategy switch. Reuse one service context across claims to preserve pacing and OpenAIRE's hourly request budget. Environment keys/flags are loaded; OpenAIRE is enabled for this new policy unless explicitly disabled. Legacy defaults are untouched.
- [`arxiv_fallback.py`](../../src/paper_search/arxiv_fallback.py): **off by default**. Trigger is fewer than `min(5, top_k)` unique candidates, never an API error. Maximum two logical fallbacks, stopping once enough candidates are accumulated. Explicitly annotated optional phrases are removed last-first; then phrase-order constraints may be relaxed while retaining every required scientific token, acronym, number and symbol. Date bounds never change. Exact wire requests, omitted phrases and intentional constraint relaxation are logged in result diagnostics. Primary candidates are retained before fallback candidates within per-provider depth.
- All existing spaCy-selected phrases are required by default. The code does **not** guess which scientific condition/outcome can safely be dropped. Optional annotations must describe the actual primary query. Thus the frozen ten-claim plan only relaxes phrase order where possible; it cannot rescue every zero or guarantee that the existing extractor retained every original outcome. Original claim and extraction omissions remain available for review. Phrase-order relaxation can reduce precision and requires evaluation.
- [`evaluation/depth.py`](../../src/paper_search/evaluation/depth.py): response-prefix summaries deduplicate each normalized prefix and keep unknown/conflicting labels separate from irrelevant. No ranking or accuracy claim is introduced.

## 3. Label cleanup and manual review

[Audit summary](audit_summary.json) reconstructs identities in frozen historical encounter order, maps label sources by `(claim_id, canonical identity)`, verifies the saved manifest mapping, and records input SHA-256 hashes. Across **450 historical pairs**: **336 consistent provisional labels, 12 conflicts, 102 blank**. Thus 114 pairs cannot be treated as labelled; the blank group comprises 94 OpenAIRE, six remaining bulk pairs and two new arXiv pairs. There is no automatic source-priority overwrite.

Conflicts exported with titles, abstracts, all label values and source paths in [conflicting_labels_12.csv](conflicting_labels_12.csv): `c01:59`, `c02:7`, `c03:86`, `c03:98`, `c04:105`, `c05:15`, `c05:16`, `c06:149`, `c08:22`, `c09:202`, `c10:30`, `c10:43`. All 94 unresolved earlier OpenAIRE additions are in [unresolved_openaire.csv](unresolved_openaire.csv).

Review [manual_validation_47.csv](manual_validation_47.csv): **12 conflicts + 18 unresolved OpenAIRE + 15 consensus checks (five per provisional label) + two new arXiv pairs**. Every claim appears at least three times. Reviewers see only an opaque ID, original claim, title, abstract and blank review fields; provider, method, rank and previous labels are hidden. Abstracts exceeding 12,000 characters are explicitly flagged as truncated; consult the separate audit/full export if needed. Keep [manual_validation_audit.csv](manual_validation_audit.csv) hidden during review.

Use 0 = irrelevant, 1 = topical, 2 = potentially useful evidence for the original claim. Prefer two independent reviewers and adjudicate disagreements, especially all 12 conflicts. Leave uncertain judgments blank with notes. The sample deliberately oversamples disputes/unknowns and is **not an unbiased estimate of overall relevance accuracy**. No human labels have been generated or replaced. Preparation refuses to overwrite an existing output directory.

## 4. Prepared experiments — NOT RUN; require your approval

| Experiment | Frozen plan / maximum cost | Analysis after running |
|---|---|---|
| Retrieval depth | [depth_plan.json](depth_plan.json): 10 claims × 3 current-policy providers × one top-30 request = **30 logical/HTTP attempts maximum**, zero retries, fallback off; dates 2020-01-01–2026-10-30 | Evaluate nested normalized response-order prefixes 10/20/30; report unique candidates, known relevant/evidence yield, unknowns, abstract/PDF metadata and new assessment needs. Additional requests between prefixes = **0**. |
| arXiv fallback | [fallback_plan.json](fallback_plan.json): ten top-10 primary queries; bounded conditional fallbacks; **18 attempts maximum for these frozen concepts**, zero retries | Compare each primary with its primary-plus-fallback union; exact queries/omissions, zero/low pools, latency and additional request costs. Preserve unknown relevance. No separate duplicated control requests. |

[`run_readiness_experiment.py`](../../scripts/run_readiness_experiment.py) freezes queries from saved experiments, archives response bodies/parameters and per-claim results immediately, and stops a provider after a failure. Its experimental zero-retry budget is deliberate; **production transport retries are unchanged**. It requires `--run-live` and a fresh output directory. It does not resume or silently replay partial runs. These commands are documented for later approval, **not executed now**:

```powershell
.venv\Scripts\python.exe scripts/run_readiness_experiment.py depth --plan artifacts/integration-readiness-20261101/depth_plan.json --output artifacts/depth-approved-run --run-live
.venv\Scripts\python.exe scripts/run_readiness_experiment.py fallback --plan artifacts/integration-readiness-20261101/fallback_plan.json --output artifacts/fallback-approved-run --run-live
```

A single top-30 fetch cannot measure separate top-10/top-20 request latency. The runner reports **shared top-30 latency/cost** and unassessed prefix coverage; its output is not completed relevance evaluation. Reuse known nonconflicting labels by conservative paper identity before calling `prefix_summary(..., label_for)` offline, then assess novel candidates. Do not treat all new candidates as irrelevant or report an unassessed yield as zero true relevance. A separate repeated-depth timing study would require additional approval and requests. API order/coverage can vary from historical top-10 runs; compare prefixes within the same future fetch. PDF URLs remain metadata only.

## 5. Integration readiness and next actions

The additive service is ready for **offline Person 2 contract integration**, with mocked provider failures and provenance/dedup tests. Example and lifecycle details are in [docs/retrieval_integration.md](../../docs/retrieval_integration.md). No live end-to-end validation of this new service has been performed. Required local dependencies remain the existing pinned MiniLM weights and spaCy `en_core_web_sm`; no new models/providers were added.

Next: manually review the 47 pairs, reconcile conflicts without overwriting sources, approve the two bounded experiments if desired, and assess new deeper/fallback candidates before selecting depth or enabling relaxation. Keep arXiv fallback disabled in production until that evidence exists. Proceed with Person 2's offline handoff meanwhile.

Validation: **261 pytest tests passed; Ruff passed** (offline, including the existing suite). No retrieval APIs, LLM evaluation, PDF verification or ranking were run or implemented.
