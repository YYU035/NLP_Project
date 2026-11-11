"""Person 5 research workspace: discovery, verification and explanation review."""
from pathlib import Path
import sys
import streamlit as st
from research_verifier.ui.discovery import render_discovery
from research_verifier.ui.verification import render_verification
from research_verifier.ui.reviews import render_reviews
from research_verifier.ui.benchmarks import render_benchmarks
from research_verifier.ui.attribution import render_attribution
from research_verifier.ui.pdf import render_pdf
from research_verifier.ui.overview import render_overview

st.set_page_config(page_title="Evidence workspace", page_icon="📖", layout="wide")
# Streamlit removes unmounted widget state when changing workspaces. Keep a small
# allowlisted input memory so returning cannot associate a result with a reset claim.
INPUT_KEYS = ["mode", "scenario", "claim", "benchmark_dataset", "pdf_claim", "attribution_baseline", "attribution_method", "discovery_source", "discovery_claim", "discovery_top_k",
              "discovery_from", "discovery_to", "discovery_filter", "discovery_provider_filter", "discovery_pdf_only",
              "discovery_page", "discovery_inspect", "reviewer_id", "review_output", "review_confirm",
              *["review_" + name for name in ("factual_supported", "factual_total", "citation_supported", "citation_total",
                 "citation_complete", "citation_required", "verdict_consistent", "scope_preserved",
                 "correction_supported", "correction_total", "usefulness", "notes")]]
st.session_state.setdefault('workspace','Find papers' if 'discovery_snapshot' in st.session_state else 'Project overview')
memory = st.session_state.setdefault("input_memory", {})
for key in INPUT_KEYS:
    if key in st.session_state:
        memory[key] = st.session_state[key]
    elif key in memory:
        st.session_state[key] = memory[key]
with st.sidebar:
    st.title("Research evidence")
    active = any(key in st.session_state for key in ("discovery_job", "pipeline_job", "aggregation_job", "llm_job", "attribution_job", "pdf_job"))
    workspace = st.radio("Workspace", ["Find papers", "Verify evidence", "Local PDF", "Evaluate explanations", "Benchmark results", "Model attribution", "Project overview"], key="workspace", disabled=active)
    st.divider()
if workspace == "Find papers":
    render_discovery()
elif workspace == "Verify evidence":
    render_verification()
elif workspace == 'Benchmark results':
    render_benchmarks()
elif workspace == 'Model attribution':
    render_attribution()
elif workspace == 'Project overview':
    render_overview()
elif workspace == 'Local PDF':
    render_pdf()
else:
    render_reviews()
for key in INPUT_KEYS:
    if key in st.session_state:
        memory[key] = st.session_state[key]
