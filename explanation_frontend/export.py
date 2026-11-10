"""Versioned, self-contained JSON replay; no pickle, code loading or CSV export."""
from __future__ import annotations

from typing import Literal

from .models import AggregationResponse, Explanation, ExplanationContext, Model, RunSnapshot, digest, make_context
from .validation import validate_explanation


class ReplayBundle(Model):
    export_version: Literal["1.0"] = "1.0"
    replay_mode: Literal["cached"] = "cached"
    snapshot: RunSnapshot
    snapshot_hash: str
    selected_paper_ids: list[str]
    aggregation: AggregationResponse
    explanation: Explanation


def export_snapshot(context: ExplanationContext, explanation: Explanation) -> str:
    if not validate_explanation(explanation, context).displayable:
        raise ValueError("Cannot export an invalid explanation")
    return ReplayBundle(snapshot=context.snapshot, snapshot_hash=context.snapshot.fingerprint,
                        selected_paper_ids=context.selected_paper_ids, aggregation=context.aggregation,
                        explanation=explanation).model_dump_json(indent=2)


def import_snapshot(raw: str | bytes) -> ReplayBundle:
    if len(raw) > 10_000_000:
        raise ValueError("Replay file exceeds 10 MB")
    bundle = ReplayBundle.model_validate_json(raw)
    if bundle.snapshot_hash != bundle.snapshot.fingerprint:
        raise ValueError("Snapshot checksum mismatch")
    context = make_context(bundle.snapshot, bundle.selected_paper_ids, bundle.aggregation)
    # Early integrated exports predate this one safety note. Restore only the
    # exact known omission; do not accept arbitrary user-supplied limitations.
    from .templates import build_template, AUTOMATIC_SELECTION_LIMITATION, LOCAL_PDF_LIMITATION
    expected = build_template(context).limitations
    migrated_notes = {AUTOMATIC_SELECTION_LIMITATION, LOCAL_PDF_LIMITATION}
    legacy = [note for note in expected if note not in migrated_notes]
    if any(note in expected for note in migrated_notes) and bundle.explanation.limitations == legacy:
        bundle = bundle.model_copy(update={"explanation": bundle.explanation.model_copy(
            update={"limitations": expected})})
    if not validate_explanation(bundle.explanation, context).displayable:
        raise ValueError("Replay explanation fails grounding checks")
    return bundle
