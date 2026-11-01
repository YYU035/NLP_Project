"""Coverage comparisons with explicit failure and missing-label semantics."""

from typing import Any

from paper_search.deduplication import group_papers
from paper_search.evaluation.models import SearchObservation
from paper_search.metadata import ScientificPaper


def compare_paper_sets(
    original: list[ScientificPaper],
    reformulated: list[ScientificPaper],
) -> dict[str, Any]:
    """Compare identity groups and retain all source records."""
    grouped = group_papers(original + reformulated)
    left = set(grouped.assignments[: len(original)])
    right = set(grouped.assignments[len(original) :])
    return {
        "original_unique": len(left),
        "reformulated_unique": len(right),
        "overlap": len(left & right),
        "additional": len(right - left),
        "lost": len(left - right),
        "jaccard": len(left & right) / len(left | right)
        if left | right
        else 1,
        "original_group_ids": sorted(left),
        "reformulated_group_ids": sorted(right),
        "groups": [
            [paper.model_dump(mode="json") for paper in group]
            for group in grouped.groups
        ],
    }


def compare_calls(
    calls: list[SearchObservation], *, matched: bool = False
) -> dict[str, Any]:
    """Compare natural cost or repeat-original matched logical-request budgets.

    The original response is shared by both arms. Matched budget repeats the
    original live when an alternative exists, rather than fabricating a second
    result list from a cache. Retry counts can make wire budgets unequal.
    """
    original = [c for c in calls if c.kind == "original"]
    alternatives = [c for c in calls if c.kind == "alternative"]
    repeats = [c for c in calls if c.kind == "original_repeat"]
    left = original + (repeats if matched else [])
    right = original + alternatives
    left_papers = [paper for call in left for paper in call.papers]
    right_papers = [
        paper for call in right for paper in call.papers
    ]
    complete = bool(original) and all(
        c.succeeded for c in left + right
    )
    measured = compare_paper_sets(left_papers, right_papers)
    counts = {
        key: measured[key] if complete else None
        for key in (
            "original_unique",
            "reformulated_unique",
            "overlap",
            "additional",
            "lost",
            "jaccard",
        )
    }
    left_requests = sum(len(c.attempts) for c in left)
    right_requests = sum(len(c.attempts) for c in right)
    return {
        **counts,
        "status": "complete"
        if complete
        else "incomplete_provider_failure",
        "observed_original_unique": measured["original_unique"],
        "observed_reformulated_unique": measured[
            "reformulated_unique"
        ],
        "original_zero": not left_papers
        if all(c.succeeded for c in left) and left
        else None,
        "reformulated_zero": not right_papers
        if all(c.succeeded for c in right) and right
        else None,
        "original_logical_requests": len(left),
        "reformulated_logical_requests": len(right),
        "original_api_requests": left_requests,
        "reformulated_api_requests": right_requests,
        "logical_budget_matched": len(left) == len(right),
        "wire_budget_matched": left_requests == right_requests,
        "original_latency_seconds": sum(
            c.elapsed_seconds for c in left
        ),
        "reformulated_latency_seconds": sum(
            c.elapsed_seconds for c in right
        ),
        "extra_query_api_requests": sum(
            len(c.attempts) for c in alternatives
        ),
        "extra_query_latency_seconds": sum(
            c.elapsed_seconds for c in alternatives
        ),
        "repeat_original_overhead_requests": sum(
            len(c.attempts) for c in repeats
        ),
        "original_call_ids": [c.id for c in left],
        "reformulated_call_ids": [c.id for c in right],
        "identity_comparison": measured,
        "precision_at_k": None,
        "recall_at_k": None,
        "relevance_labels": "unavailable; coverage only",
    }


def summarize_provider(
    calls: list[SearchObservation],
) -> dict[str, Any]:
    """Summarize coverage, metadata, failures, and paired comparisons."""
    successes = [c for c in calls if c.succeeded]
    papers = [p for c in successes for p in c.papers]
    natural = [c for c in calls if c.kind != "original_repeat"]
    originals = [c for c in calls if c.kind == "original"]
    cases = sorted({c.claim_id for c in calls})
    comparisons = [
        compare_calls([c for c in calls if c.claim_id == case])
        for case in cases
    ]
    complete = [
        comparison
        for comparison in comparisons
        if comparison["status"] == "complete"
    ]
    measured = compare_paper_sets(
        [p for c in originals for p in c.papers],
        [p for c in natural for p in c.papers],
    )
    return {
        "logical_calls": len(calls),
        "successful_calls": len(successes),
        "failures": len(calls) - len(successes),
        "wire_requests": sum(len(c.attempts) for c in calls),
        "http_429_responses": sum(
            a.get("status") == 429
            for c in calls
            for a in c.attempts
        ),
        "zero_result_calls": sum(
            c.status == "zero_results" for c in calls
        ),
        "zero_result_rate_successful_calls": (
            sum(c.status == "zero_results" for c in successes)
            / len(successes)
            if successes
            else None
        ),
        "original_zero_claims": sum(
            x["original_zero"] is True for x in comparisons
        ),
        "original_zero_denominator": sum(
            x["original_zero"] is not None for x in comparisons
        ),
        "reformulated_zero_claims": sum(
            x["reformulated_zero"] is True for x in comparisons
        ),
        "reformulated_zero_denominator": sum(
            x["reformulated_zero"] is not None for x in comparisons
        ),
        "complete_claim_comparisons": len(complete),
        "coverage_gain_claims": sum(
            (x["additional"] or 0) > 0 for x in complete
        ),
        "paired_additional_sum": sum(
            x["additional"] for x in complete
        ),
        "observed_original_unique": measured["original_unique"],
        "observed_reformulated_unique": measured[
            "reformulated_unique"
        ],
        "observed_additional_unique": measured["additional"],
        "all_natural_calls_successful": all(
            c.succeeded for c in natural
        ),
        "natural_wire_requests": sum(
            len(c.attempts) for c in natural
        ),
        "natural_successful_calls": sum(c.succeeded for c in natural),
        "extra_query_wire_requests": sum(
            len(c.attempts)
            for c in calls
            if c.kind == "alternative"
        ),
        "matched_repeat_overhead_requests": sum(
            len(c.attempts)
            for c in calls
            if c.kind == "original_repeat"
        ),
        "latency_seconds": sum(c.elapsed_seconds for c in calls),
        "metadata_record_count": len(papers),
        "metadata_present": {
            name: sum(bool(getattr(p, name)) for p in papers)
            for name in (
                "title",
                "authors",
                "doi",
                "abstract",
                "publication_date",
                "landing_url",
                "pdf_urls",
            )
        },
        "date_filter_violations": sum(
            c.metadata.get("date_filter_violations", 0)
            for c in successes
            if c.metadata
        ),
        "original_metadata_record_count": sum(
            len(c.papers) for c in originals
        ),
        "original_metadata_present": {
            name: sum(
                bool(getattr(p, name))
                for c in originals
                for p in c.papers
            )
            for name in (
                "title",
                "authors",
                "doi",
                "abstract",
                "publication_date",
                "landing_url",
                "pdf_urls",
            )
        },
        "original_successes": sum(c.succeeded for c in originals),
        "original_failures": sum(
            not c.succeeded for c in originals
        ),
        "original_wire_requests": sum(
            len(c.attempts) for c in originals
        ),
        "original_latency_seconds": sum(
            c.elapsed_seconds for c in originals
        ),
    }
