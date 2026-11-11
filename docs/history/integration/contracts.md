# Current integration status - 6 November 2026

The built-in factory explanation_frontend.pipeline:create_provider now connects real discovery, local Naila BM25/SBERT, Yu Yue RRF, frontend exact-abstract selection, Chong Yao offline NLI and cached aggregation. Earlier notes about missing callables below describe initial development boundaries; the current additions and policy ledger is INTEGRATION_CHANGELOG.md. Frontend source folders remain isolated; teammate runtimes are pinned copies/dependencies.

The missing passage stage is filled by an explicitly provisional exact-span selector. The frontend conflict policy preserves both sides of internal/across-paper conflicts. These additions need team acceptance and must not be represented as unchanged owner algorithms. Original claim/IDs, source metadata, exact text/offsets, model revisions and full ranking output are retained in snapshots. Pipeline rank output is also serialized in filters.ranking_output_json for runs containing unranked no-text candidates. Built-in adapters use the same local RunSnapshot schema and citation validators described below.

# Person 5 handoff contract — local schema 1.0

The authoritative team requirements are the Project Role/Task Split and Edge Cases tabs. These local types are proposed integration mappings, not a claim that shared/ has implemented them. The complete machine-readable types are in models.py. fixtures/conflict.json is a full example; all fixtures are synthetic.

## Owner boundary

1. Person 1: paper IDs, metadata, exact original claim, access/version/status flags, filters, retrieval time and search completeness.
2. Person 2: ranked paper IDs and explicitly typed retrieval scores.
3. Person 3: exact selected passages, stable IDs and source references, with preserved qualifications and context.
4. Person 4: `verify(claim, papers, passages_by_paper)` and `aggregate(verification_response, subset_of_paper_ids)`; the second operation must not repeat inference. Supply per-passage labels, per-paper verdicts/conflict flags/causes, processing failures and model settings.
5. Person 5: schema validation, request state, display, explanation, citation checking, export/replay and explanation evaluation. This module never infers missing upstream scientific labels.

The older Project Ideas signatures and fixed 0.70 threshold are not used. Thresholds arrive from the calibrated verifier. Corpus verdicts include CONFLICTING. Within-paper conflict uses conflict_flag. Processing failure is an operational status, represented with null corpus verdict; it is not scientific INSUFFICIENT.

## Models and constraints

`RunSnapshot` includes schema_version="1.0", run_id, exact claim, timezone-aware created_at/retrieved_at, mode, source_snapshot_id, filters, search_complete, papers, verification, provisional, stage_statuses and optional explicit_replacements.

`Paper`: paper_id, title, authors, optional year/venue/url/version/access, source_status (default unknown), has_fulltext, retrieval_scores keyed by method. Missing metadata is not invented. The verifier's evidence_type says what was actually inspected, which can differ from full-text availability.

`EvidencePassage`: passage_id, paper_id, section, exact text, optional score, page, sentence_indices and source_version. IDs must be unique and must link to a supplied paper. Sentence indices are upstream values (document whether zero-based); pages are positive, one-based PDF page numbers if supplied. No page is fabricated for an abstract.

`VerificationResponse`: per_paper_verdicts and per_passage_labels maps, evidence_passages list, processing_failures list and model_settings. Failure records deliberately contain stage/code/paper_id, not exception bodies. Normalize upstream failures to safe codes in the bridge.

`PaperVerdict`: paper_id, title, verdict in SUPPORTED/CONTRADICTED/INSUFFICIENT, optional confidence, evidence_type abstract_only/full_text, sufficiency_cause, conflict_flag, supporting_passage_ids, contradicting_passage_ids. Every cited ID must exist and have the matching passage label and paper.

`AggregationResponse`: original claim, aggregated_verdict (four scientific verdicts or null on failure), exact primary_evidence, optional global_confidence, verdict_delta_reason, status ok/partial/failed and optional sufficiency_cause. INSUFFICIENT requires a cause. Decisive verdicts require corresponding SUPPORT/CONTRADICT primary passages. Neutral or truncated passages remain inspectable but cannot be primary explanation evidence.

`ModelSettings`: checkpoint, revision, max_length, optional calibrated threshold and seed. Scores must be finite and confidence in [0,1]. They are model outputs, not truth probabilities.

`explicit_replacements`: optional passage_id → WHOLE exact passage text. The upstream bridge must populate it only after explicitly establishing that the passage is a supported replacement assertion for this exact claim. Person 5 does not infer this property or populate it in demos. If absent, corrections are withheld.

Unknown fields fail validation, forcing an intentional adapter mapping. Convert upstream alternate field names and list/map representations in your factory/pack function. Do not drop status, source versions or scientific uncertainty just to pass validation.

## Injected live adapter

`IntegratedProvider(search=..., rank=..., select=..., verify=..., aggregate=..., pack=...)` connects real callables:

```python
# These are integration boundaries; replace the functions with the owners' actual code.
provider = IntegratedProvider(
    search=search,       # (original_claim, options) -> paper records
    rank=rank,           # (original_claim, papers) -> ordered paper records
    select=select,       # (original_claim, ranked_papers) -> passages_by_paper
    verify=verify,       # (original_claim, ranked_papers, passages_by_paper) -> original response
    aggregate=aggregate, # (same original response, selected_IDs) -> mapped AggregationResponse
    pack=pack,           # (claim, options, papers, response, aggregation) -> local RunSnapshot/dict
)
```

`run()` caches the original typed verifier response in this provider instance, tied to the validated snapshot fingerprint. Subsequent `aggregate()` calls receive that same response, preserving upstream custom fields. A different snapshot fails explicitly. Use a new provider instance for every session/run; never share the mutable provider as a global singleton. Upstream functions must apply network/model timeouts and avoid mutating the cached response.

The server environment variable `EXPLANATION_PIPELINE_FACTORY=your_bridge:create_provider` names a trusted factory returning this provider. A future bridge can live inside explanation_frontend without changing app.py. The application will not guess teammate import paths, execute uploaded Python, or silently use fixtures when live configuration fails.

## Replay policy

JSON export contains the full snapshot and SHA-256 checksum, selected IDs, selected aggregation and final displayable explanation. Import validates the schema, exact claim, checksum, citation scope and grounding. The checksum detects accidental modification, not forgery or source authenticity. Original demo/live mode remains in the snapshot; the interface separately labels CACHED REPLAY.

An unchanged replay selection uses its saved selected aggregation exactly. A changed selection uses the explicit local replay policy (presence of supporting/contradicting paper verdicts, conflict flags, max score); it is labelled and is not a claim of equivalence with Person 4's published algorithm. Live mode always delegates changed selections to Person 4.

## Explanation policy

The verifier's verdict is authoritative. Template explanations quote whole eligible primary passages. LLM statements must match whole allowed passages exactly, retain both conflict sides, and match the selected snapshot/aggregation context. Generated free-form text, fabricated references, partial quotations and unapproved corrections fail closed to the template. The current scope is intentionally extractive; semantic citation support still needs independent human review. No automated validator here proves the underlying paper true, relevant, authentic or scientifically sound.

## Discovery handoff now implemented

discovery.py validates the actual Person 1 RetrievalResult schema at Raghavi revision 259d2a9. It preserves normalized source records, identity aliases and provider/query origins. DiscoverySnapshot is independent of RunSnapshot: candidates have no scientific verdict. Native payloads and checksum-protected discovery exports are accepted; absent retrieval dates remain unknown. Query reformulations never replace original_claim.

LiveDiscoveryBridge keeps one async service/event loop per session, with explicit close and deadlines. It injects the pinned local KeyBERT model path, leaves arXiv fallback disabled and preserves upstream provider policy. Preparing selected candidates for verification does not repeat search.

FromDiscoveryProvider is the downstream-only seam: verify_selected receives discovery-selection-1.0 (original_claim, selected full candidate entries and provider diagnostics) and returns a mapped RunSnapshot. Owners must retain the canonical identities. aggregate receives that cached snapshot plus selected IDs. The adapter checks claim/selection lineage and attaches optional discovery_context, preserving source data and original selection. Cached discovery stays cached in its provenance even when verification is newly computed.

Original verification replay hashes remain compatible when discovery_context is absent. EXPLANATION_VERIFY_DISCOVERY_FACTORY configures the selected-search bridge separately. Until it is supplied, the UI reports missing integration rather than fabricating scientific verdicts.

## Still to settle with owners

- Persons 2–4 import paths and exact downstream mapping; fixed benchmark split/version. Person 1 discovery is connected through its pinned dependency.
- Source ID/version conventions, zero/one-based sentence indices, retraction/eligibility handling and available safe status codes.
- Team review of the frontend conflict policy and broader cached latency/held-out evaluation; checkpoint and threshold are now explicitly pinned in the built-in pipeline.
- Actual replacement-evidence approval method, if corrections are included.
- The downloadable detailed edge-case checklist, whose individual trigger/priority mapping was not present in the supplied tab.

## Ranking context extension

DiscoverySnapshot has optional `ranking_context` containing the unchanged Person 3 RankingOutput schema at a91e257. Absent context preserves legacy snapshot fingerprints. Connected inputs must match the entire loaded DiscoveryResult; RRF rows are recomputed to validate scores, ranks, identities and metadata. Reranking is currently rejected. Only output IDs may be selected. The selected-paper handoff adds nullable `ranking_context` and orders full selected records by final rank. Consumers should accept that additive field; strict downstream parsers will need their owners to update. Existing verification replay retains this context inside discovery_context. See RANKING_INTEGRATION.md for operational details.

## Published Person 4 contract (5 November 2026)

The claim_verifier module is now available on chong_yao_claim_verification at 1ac06b675e3d0719a63cb8efe98eda4f2c5fd576. Its verify/aggregate APIs match the planned two-stage boundary but are not drop-in frontend schemas. Lists must be converted to checked ID maps; abstract maps to abstract_only; tokenizer_max_length maps to max_length. Operational failures and insufficiency causes require explicit mapping. Keep the original upstream VerificationResponse cached for aggregate calls. Packaging, immutable model revision/offline loading, conflict_flag behavior and preservation of both sides in capped primary evidence still require validation. See README.md for the current owner checklist and source findings. No real verifier execution is claimed by this documentation update.

Person4Adapter now implements the published response mapping and caches the native typed response for the actual aggregator. Pinned code is isolated under _upstream/claim_verifier with a relative schema import. Synthetic adapter/owner-aggregation tests pass; internal conflicts fail closed pending owner resolution. The real neural verifier and selected-search factory are not yet enabled. See PERSON4_INTEGRATION.md.

## QC compatibility clarification - 6 November 2026

Version-1 integrated exports written before the automatic abstract-source safety note are accepted only when their limitation list exactly matches that known earlier canonical list. Import restores the missing note and runs the complete citation/verdict/claim gate. It never changes snapshot bytes/hash, verdict or source text; arbitrary limitation changes are rejected. Model/HTML/CSV safety and structured failure rules are unchanged. Current task document confirms Person 3 owns passage selection; TEAM_QC_REPORT.md records contract findings and next steps for all owners.
