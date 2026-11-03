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

An unchanged replay selection uses its saved selected aggregation exactly. A changed selection uses the explicit local replay policy (presence of supporting/contradicting paper verdicts, conflict flags, max score); it is labelled and is not a claim of equivalence with Person 4's future algorithm. Live mode always delegates changed selections to Person 4.

## Explanation policy

The verifier's verdict is authoritative. Template explanations quote whole eligible primary passages. LLM statements must match whole allowed passages exactly, retain both conflict sides, and match the selected snapshot/aggregation context. Generated free-form text, fabricated references, partial quotations and unapproved corrections fail closed to the template. The current scope is intentionally extractive; semantic citation support still needs independent human review. No automated validator here proves the underlying paper true, relevant, authentic or scientifically sound.

## Still to settle with owners

- Final import paths and exact mapping into these local schemas; fixed benchmark split/version.
- Source ID/version conventions, zero/one-based sentence indices, retraction/eligibility handling and available safe status codes.
- Person 4 aggregation rules and latency on real cached outputs; threshold/checkpoint revision.
- Actual replacement-evidence approval method, if corrections are included.
- The downloadable detailed edge-case checklist, whose individual trigger/priority mapping was not present in the supplied tab.
