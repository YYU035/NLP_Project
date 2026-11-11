"""Shared advisory presentation for stored abstracts, independent of NLI."""
import streamlit as st

from .abstract_quality import (
    ADVISORY, QUALITY_LABELS, AbstractQualityReview, quality_label, quality_reasons,
)
from ._upstream.discovery_quality.abstract_quality import AbstractQualityAssessment


def render_abstract_quality(assessment: AbstractQualityAssessment):
    label = "Abstract check: " + quality_label(assessment)
    if assessment.status == "likely_valid":
        st.caption(label)
    else:
        st.warning(label)
    for reason in quality_reasons(assessment):
        st.caption(reason)
    st.caption(ADVISORY)


def render_quality_summary(review: AbstractQualityReview, *, key_prefix: str):
    counts = review.counts()
    total = sum(counts.values())
    flagged = total - counts.get("likely_valid", 0)
    st.subheader("Abstract quality checks")
    if total:
        st.caption(
            f"All stored candidates: {total} descriptions; {flagged} need review "
            f"({counts.get('missing', 0)} missing). Counts are independent of the current view filters."
        )
    else:
        st.caption("No candidate abstracts to assess.")
    with st.expander("About the abstract checks and download"):
        st.caption(ADVISORY)
        st.caption("Provider occurrences and unique candidates are counted separately in the report.")
        if review.provider_error:
            st.caption(review.provider_error)
        if counts:
            st.table([{"Abstract check": QUALITY_LABELS[status], "Candidates": count}
                      for status, count in counts.items()])
        st.download_button(
            "Download abstract quality report", review.to_json(),
            "abstract-quality-review.json", "application/json",
            key=f"{key_prefix}:abstract-quality:{review.source_fingerprint}",
        )
