"""Read-only abstract diagnostics over existing discovery contracts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json

from research_verifier.discovery.service import DiscoverySnapshot
from research_verifier.discovery.quality_rules import RULE_VERSION, AbstractQualityAssessment, AbstractQualityRecord, assess_abstract, assess_retrieval

QUALITY_SOURCE_REVISION = "0ee95651cdd0c9d40a0ffb09a6bba896db3be455"
ADVISORY = (
    "Advisory only: these rules do not establish abstract completeness, "
    "relevance or scientific accuracy. Text, paper selection, ranking and "
    "verification are unchanged."
)
QUALITY_LABELS = {
    "missing": "Missing abstract",
    "body_like": "Review: possibly article body",
    "suspected_incomplete": "Review: possible cut-off",
    "uncertain": "Review: uncertain description",
    "likely_valid": "No warning detected",
}
QUALITY_REASONS = {
    "empty_or_absent": "No abstract or description was supplied.",
    "excessive_length": "More than 1,000 words; this may be article body text or a long abstract.",
    "multiple_body_section_signals": "Long text includes multiple article-body section signals.",
    "continuation_with_unresolved_reference": "The opening appears to continue an earlier sentence.",
    "explicit_cutoff_with_dangling_clause": "The text ends with an ellipsis after an unfinished phrase.",
    "placeholder_text": "The supplied text appears to be an unavailable-abstract placeholder.",
    "possible_repeated_translations": "Repeated abstract or translation headings need review.",
    "possible_cutoff": "A trailing ellipsis may indicate a cut-off.",
    "replacement_character": "A replacement character may indicate a text-encoding problem.",
    "sentence_structure_unclear": "The rules could not identify a clear sentence boundary.",
    "no_detected_warning_signs": "No warning was found by these limited rules.",
}


def quality_label(assessment: AbstractQualityAssessment) -> str:
    return QUALITY_LABELS[assessment.status]


def quality_reasons(assessment: AbstractQualityAssessment) -> list[str]:
    return [QUALITY_REASONS[reason] for reason in assessment.reason_codes]


@dataclass(frozen=True)
class AbstractQualityReview:
    source_fingerprint: str
    source_reference: str
    original_claim: str
    records: tuple[AbstractQualityRecord, ...]
    provider_error: str | None = None

    @property
    def by_identity(self) -> dict[str, AbstractQualityAssessment]:
        return {row.identity: row.assessment for row in self.records
                if row.scope == "deduplicated"}

    def counts(self, scope: str = "deduplicated") -> dict[str, int]:
        if scope not in {"deduplicated", "provider_record"}:
            raise ValueError("Unknown abstract-quality scope")
        return dict(Counter(row.assessment.status for row in self.records
                            if row.scope == scope))

    def to_json(self) -> str:
        return json.dumps({
            "report_version": "1.0",
            "advisory_only": True,
            "assessment_mode": "offline_current_rules_on_stored_text",
            "rule_version": RULE_VERSION,
            "source_revision": QUALITY_SOURCE_REVISION,
            "source_fingerprint": self.source_fingerprint,
            "source_reference": self.source_reference,
            "original_claim": self.original_claim,
            "limitations": ADVISORY,
            "scope_units": {
                "deduplicated": "Unique claim-paper pairs in this search",
                "provider_record": "Provider occurrences, including repeated query hits",
            },
            "scope_counts": {scope: self.counts(scope)
                             for scope in ("deduplicated", "provider_record")},
            "provider_records_complete": self.provider_error is None,
            "provider_error": self.provider_error,
            "assessments": [row.model_dump(mode="json") for row in self.records],
        }, ensure_ascii=False, indent=2)


def review_discovery(snapshot: DiscoverySnapshot) -> AbstractQualityReview:
    """Assess without changing the search, source text, order or fingerprint.

    Legacy handoffs can omit provider records or their reverse provenance links.
    Keep candidate checks usable and disclose incomplete occurrence reporting.
    """
    provider_error = None
    fingerprint = snapshot.fingerprint
    expected_origins = {
        (origin.provider, origin.source_id, origin.query_id, origin.rank)
        for entry in snapshot.result.papers for origin in entry.provenance
    }
    stored_origins = {
        (provider, record.paper.source_id, record.query_id, record.rank)
        for provider, run in snapshot.result.providers.items()
        for record in run.records
    }
    if expected_origins - stored_origins:
        provider_error = (
            "Provider-occurrence reporting is incomplete: some retained "
            "provenance links have no stored provider record. Counts cover "
            "stored occurrences only. Candidate abstract checks remain available."
        )
    try:
        records = assess_retrieval(snapshot.result, claim_id=fingerprint)
    except ValueError:
        provider_error = (
            "Provider-occurrence checks are unavailable because some source "
            "records do not resolve to retained candidate identities. "
            "Candidate abstract checks remain available."
        )
        records = [AbstractQualityRecord(
            claim_id=fingerprint,
            original_claim=snapshot.result.original_claim,
            scope="deduplicated",
            identity=entry.identity,
            identity_aliases=list(entry.identity_aliases),
            provider=entry.paper.source,
            source_id=entry.paper.source_id,
            title=entry.paper.title,
            doi=entry.paper.doi,
            provenance=[origin.model_dump() for origin in entry.provenance],
            assessment=assess_abstract(entry.paper.abstract),
        ) for entry in snapshot.result.papers]
    return AbstractQualityReview(
        fingerprint, snapshot.source_reference,
        snapshot.result.original_claim, tuple(records), provider_error,
    )
