"""Adapt published Person 4 responses; retain the original for subset aggregation."""
from datetime import datetime, timezone

from research_verifier.contracts.models import RunSnapshot, VerificationResponse, AggregationResponse, Paper, EvidencePassage, PassageVerdict, PaperVerdict, ProcessingFailure, ModelSettings, make_context
from research_verifier.verification.schemas import VerificationResponse as NativeResponse

REVISION = "1ac06b675e3d0719a63cb8efe98eda4f2c5fd576"


class VerifierAdapter:
    """aggregate_fn must accept the original typed response and subset_of_paper_ids.

    This boundary never constructs a model or runs inference. The caller supplies
    verified paper metadata and the actual owner's response, not generated evidence.
    """
    def __init__(self, aggregate_fn):
        self.aggregate_fn = aggregate_fn
        self._native = None
        self._fingerprint = None

    def snapshot(self, response, papers: list[Paper], *, mode="cached", abstained_paper_ids=()):
        native = NativeResponse.model_validate(
            response.model_dump(mode="json") if hasattr(response, "model_dump") else response)
        paper_map = {p.paper_id: p for p in papers}
        if len(paper_map) != len(papers):
            raise ValueError("Duplicate paper IDs")
        abstained = set(abstained_paper_ids)
        if not abstained <= set(paper_map) or abstained & {p.paper_id for p in native.evidence_passages}:
            raise ValueError('Invalid selector abstention scope')
        # Successful selector abstention is not a missing-text processing failure.
        from research_verifier.verification.schemas import SufficiencyCause
        for row in native.per_paper_verdicts:
            if row.paper_id in abstained:
                row.sufficiency_cause = SufficiencyCause.NO_PASSAGE_ABOVE_THRESHOLD
        native.processing_failures = [text for text in native.processing_failures if text not in
            {f'No passages found for paper {pid}' for pid in abstained}]
        verdicts = {}
        for row in native.per_paper_verdicts:
            if row.paper_id in verdicts:
                raise ValueError("Duplicate paper verdict")
            data = row.model_dump(mode="json")
            data["evidence_type"] = {"abstract": "abstract_only", "full_text": "full_text"}[row.evidence_type]
            verdicts[row.paper_id] = PaperVerdict.model_validate(data)
        labels, broken = {}, set()
        for row in native.per_passage_labels:
            if row.passage_id in labels:
                raise ValueError("Duplicate passage label")
            if row.operational_status.value != "OK":
                broken.add(row.paper_id)
            labels[row.passage_id] = PassageVerdict(
                passage_id=row.passage_id, paper_id=row.paper_id,
                label=row.label.value if row.operational_status.value == "OK" else "NEUTRAL",
                confidence=row.confidence, truncated=row.truncated)
        if set(verdicts) != set(paper_map):
            raise ValueError("Verifier must cover exactly the supplied paper identities")
        evidence = [EvidencePassage.model_validate(p.model_dump()) for p in native.evidence_passages]
        usable = {p.paper_id for p in native.per_passage_labels
                  if p.operational_status.value == "OK" and not p.truncated}
        usable |= abstained
        failed = set(paper_map) - usable
        failures = [ProcessingFailure(stage="verification", code="NO_USABLE_PASSAGE", paper_id=pid)
                    for pid in sorted(failed | broken)]
        if native.processing_failures:
            failures.append(ProcessingFailure(stage="verification", code="UPSTREAM_REPORTED_FAILURE"))
        settings = native.model_settings
        verification = VerificationResponse(per_paper_verdicts=verdicts, per_passage_labels=labels,
            processing_failures=failures, evidence_passages=evidence,
            model_settings=ModelSettings(checkpoint=settings.checkpoint, revision=settings.revision,
                max_length=settings.tokenizer_max_length, threshold=settings.threshold, seed=settings.seed))
        # Install the typed original only after validating a complete snapshot.
        aggregation = self._aggregate(native, verification, list(paper_map))
        now = datetime.now(timezone.utc)
        snapshot = RunSnapshot(run_id=now.isoformat(), claim=native.claim, created_at=now,
            retrieved_at=now, mode=mode, source_snapshot_id=f"person4:{REVISION}",
            filters={"retrieval_timestamp": "not_supplied; retrieved_at records import time"},
            search_complete=False, papers=paper_map, verification=verification, provisional=aggregation,
            stage_statuses={"verification": "imported", "discovery": "not_supplied"})
        self._native, self._fingerprint = native, snapshot.fingerprint
        return snapshot

    def _aggregate(self, native, verification, ids):
        selected = set(ids)
        usable = {v.paper_id for v in verification.per_passage_labels.values()
                  if not v.truncated and v.label in {"SUPPORT", "CONTRADICT", "NEUTRAL"}}
        # A paper needs at least one operationally OK native label.
        usable &= {v.paper_id for v in native.per_passage_labels if v.operational_status.value == "OK"}
        usable |= {pid for pid, row in verification.per_paper_verdicts.items()
                   if row.sufficiency_cause == 'NO_PASSAGE_ABOVE_THRESHOLD'
                   and pid not in {v.paper_id for v in native.per_passage_labels}}
        failed = bool(selected) and not (selected & usable)
        if failed:
            return AggregationResponse(claim=native.claim, aggregated_verdict=None, status="failed",
                sufficiency_cause="NO_USABLE_PASSAGE", verdict_delta_reason="Verification has no usable passages.")
        output = self.aggregate_fn(native, subset_of_paper_ids=sorted(selected))
        data = output.model_dump(mode="json") if hasattr(output, "model_dump") else dict(output)
        data["verdict_delta_reason"] = data.get("verdict_delta_reason") or ""
        data["status"] = "partial" if verification.processing_failures else "ok"
        if data["aggregated_verdict"] == "INSUFFICIENT":
            data["sufficiency_cause"] = "EMPTY_SELECTION" if not selected else "NO_PASSAGE_ABOVE_THRESHOLD"
        # Do not silently reinterpret upstream internal conflicts as unanimity.
        if (any(v.conflict_flag for pid, v in verification.per_paper_verdicts.items() if pid in selected)
                and data["aggregated_verdict"] != "CONFLICTING"):
            raise ValueError("Upstream internal-conflict aggregation requires owner agreement")
        return AggregationResponse.model_validate(data)

    def aggregate(self, snapshot, paper_ids):
        if snapshot.fingerprint != self._fingerprint or self._native is None:
            raise ValueError("No matching original Person 4 response")
        if not set(paper_ids) <= set(snapshot.papers):
            raise ValueError("Unknown selected paper")
        result = self._aggregate(self._native, snapshot.verification, paper_ids)
        make_context(snapshot, paper_ids, result)
        return result
