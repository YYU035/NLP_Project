"""Mechanical grounding gate, not a claim of independent semantic verification."""
from urllib.parse import urlsplit

from research_verifier.contracts.models import Explanation, ExplanationContext, ValidationReport


def safe_url(value: str | None) -> str | None:
    if not value or any(ord(c) < 33 for c in value):
        return None
    try:
        url = urlsplit(value)
        return value if url.scheme in {"http", "https"} and url.hostname and not url.username and not url.password else None
    except ValueError:
        return None


def validate_explanation(explanation: Explanation, context: ExplanationContext) -> ValidationReport:
    from research_verifier.explanations.templates import build_template
    issues = []
    consistent = explanation.context_key == context.key and explanation.verdict == context.aggregation.aggregated_verdict
    if not consistent:
        issues.append("CONTEXT_OR_VERDICT_MISMATCH")
    if explanation.limitations != build_template(context).limitations:
        consistent = False
        issues.append("UNVALIDATED_LIMITATIONS")
    eligible = {p.passage_id: p for p in context.aggregation.primary_evidence}
    references, extractive = True, True
    statements = list(explanation.statements) + ([explanation.correction] if explanation.correction else [])
    for statement in statements:
        if any(pid not in context.passages for pid in statement.citation_ids):
            references = False
            issues.append("UNKNOWN_OR_EXCLUDED_CITATION")
        # Every cited passage must exactly equal the whole statement. This intentionally
        # rejects partial quotations/paraphrases, which can omit crucial qualifications.
        if any(pid not in eligible or eligible[pid].text != statement.text for pid in statement.citation_ids):
            extractive = False
            issues.append("NOT_WHOLE_ELIGIBLE_PASSAGE")
    if explanation.correction and any(context.snapshot.explicit_replacements.get(pid) != explanation.correction.text
                                     for pid in explanation.correction.citation_ids):
        extractive = False
        issues.append("REPLACEMENT_NOT_EXPLICITLY_SUPPLIED")
    # Require both sides of a conflict to survive generation.
    if explanation.verdict == "CONFLICTING":
        labels = {context.snapshot.verification.per_passage_labels[pid].label
                  for s in explanation.statements for pid in s.citation_ids if pid in eligible}
        if not {"SUPPORT", "CONTRADICT"} <= labels:
            extractive = False
            issues.append("CONFLICT_SIDE_OMITTED")
    if eligible and not explanation.statements:
        extractive = False
        issues.append("EVIDENCE_OMITTED")
    return ValidationReport(references_valid=references, extractive_support=extractive,
                            verdict_consistent=consistent, displayable=references and extractive and consistent,
                            issues=sorted(set(issues)))
