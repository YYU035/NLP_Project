"""Offline, advisory abstract checks; never change retrieval records."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from paper_search.retrieval_models import RetrievalResult

QualityStatus = Literal[
    "missing", "likely_valid", "suspected_incomplete", "body_like", "uncertain"
]
RULE_VERSION = "abstract-quality-v1"
LONG_TEXT_WORDS = 1000
_BODY_SECTIONS = re.compile(
    r"\b(references|acknowledg(?:e)?ments|funding|"
    r"conflicts? of interest|ethics (?:approval|committee)|"
    r"ethical approval|author contributions?)\b",
    re.IGNORECASE,
)
_TRANSLATIONS = re.compile(
    r"(?:^|[\n.!?]\s*)(?:abstract|resumen|resumo|résumé|"
    r"zusammenfassung)\s*[:\n]",
    re.IGNORECASE,
)


class AbstractQualityAssessment(BaseModel):
    """Heuristic concerns; no completeness or sufficiency guarantees."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_version: str = RULE_VERSION
    status: QualityStatus
    reason_codes: list[str] = Field(default_factory=list)
    text: str | None
    text_sha256: str | None
    character_count: int
    word_count: int


class AbstractQualityRecord(BaseModel):
    """Separate assessment linked to unchanged source and merged identities."""

    claim_id: str
    original_claim: str
    scope: Literal["provider_record", "deduplicated"]
    identity: str
    identity_aliases: list[str]
    provider: str
    source_id: str
    title: str | None
    doi: str | None
    provenance: list[dict[str, str | int]]
    assessment: AbstractQualityAssessment


def assess_abstract(text: str | None) -> AbstractQualityAssessment:
    """Inspect exact input without editing it or using a language model.

    Whitespace tokens are used only for length diagnostics. Non-ASCII symbols,
    acronyms, short length and absent final punctuation alone are not defects.
    """
    value = text.strip() if text is not None else ""
    words = len(value.split())
    reasons: list[str] = []
    if not value:
        status: QualityStatus = "missing"
        reasons.append("empty_or_absent")
    else:
        sections = set(
            m.group(0).lower() for m in _BODY_SECTIONS.finditer(value)
        )
        if words > LONG_TEXT_WORDS:
            reasons.append("excessive_length")
        if words > 300 and len(sections) >= 2:
            reasons.append("multiple_body_section_signals")
        continuation = bool(re.match(r"^(and|or|but|whereas)\b", value, re.I))
        unresolved = bool(re.search(r"\brespectively\b", value[:160], re.I))
        cutoff = bool(re.search(r"(?:\.{3}|\u2026)\s*$", value))
        dangling = bool(
            re.search(
                r"\b(and|or|of|the|with|because|which|to)\s*(?:\.{3}|\u2026)$",
                value,
                re.I,
            )
        )
        translations = len(_TRANSLATIONS.findall(value)) >= 2
        placeholder = bool(
            re.fullmatch(
                r"(?:no abstract(?: available| provided)?|"
                r"abstract unavailable|"
                r"not available|n/?a)[.!]?",
                value,
                re.I,
            )
        )
        if reasons:
            status = "body_like"
        elif continuation and unresolved:
            status = "suspected_incomplete"
            reasons.append("continuation_with_unresolved_reference")
        elif cutoff and dangling:
            status = "suspected_incomplete"
            reasons.append("explicit_cutoff_with_dangling_clause")
        elif placeholder or translations or cutoff or "\ufffd" in value:
            status = "uncertain"
        elif not re.search(r"[.!?](?:[\"')\]]|\s|$)", value):
            status = "uncertain"
            reasons.append("sentence_structure_unclear")
        else:
            status = "likely_valid"
            reasons.append("no_detected_warning_signs")
        if placeholder:
            reasons.append("placeholder_text")
        if translations:
            reasons.append("possible_repeated_translations")
        if cutoff and not dangling:
            reasons.append("possible_cutoff")
        if "\ufffd" in value:
            reasons.append("replacement_character")
    return AbstractQualityAssessment(
        status=status,
        reason_codes=reasons,
        text=text,
        text_sha256=(
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text is not None
            else None
        ),
        character_count=len(text) if text is not None else 0,
        word_count=words,
    )


def assess_retrieval(
    result: RetrievalResult, *, claim_id: str
) -> list[AbstractQualityRecord]:
    """Assess provider occurrences and selected merged texts separately.

    Provider occurrences (including repeated query hits) are not counted as
    unique papers. Existing provenance and identity aliases are copied intact.
    """
    rows = []
    origins = {
        (o.provider, o.source_id, o.query_id, o.rank): paper
        for paper in result.papers
        for o in paper.provenance
    }
    for provider, run in result.providers.items():
        for record in run.records:
            paper = record.paper
            merged = origins.get(
                (provider, paper.source_id, record.query_id, record.rank)
            )
            if merged is None:
                raise ValueError("Provider record has no merged provenance")
            rows.append(
                AbstractQualityRecord(
                    claim_id=claim_id,
                    original_claim=result.original_claim,
                    scope="provider_record",
                    identity=merged.identity,
                    identity_aliases=list(merged.identity_aliases),
                    provider=provider,
                    source_id=paper.source_id,
                    title=paper.title,
                    doi=paper.doi,
                    provenance=[
                        {
                            "provider": provider,
                            "source_id": paper.source_id,
                            "query_id": record.query_id,
                            "rank": record.rank,
                        }
                    ],
                    assessment=assess_abstract(paper.abstract),
                )
            )
    for merged in result.papers:
        paper = merged.paper
        rows.append(
            AbstractQualityRecord(
                claim_id=claim_id,
                original_claim=result.original_claim,
                scope="deduplicated",
                identity=merged.identity,
                identity_aliases=list(merged.identity_aliases),
                provider=paper.source,
                source_id=paper.source_id,
                title=paper.title,
                doi=paper.doi,
                provenance=[o.model_dump() for o in merged.provenance],
                assessment=assess_abstract(paper.abstract),
            )
        )
    return rows
