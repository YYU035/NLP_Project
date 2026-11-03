"""Person 5 research workspace: discovery, verification and explanation review."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
from explanation_frontend.discovery_view import render_discovery
from explanation_frontend.verification_view import render_verification
from explanation_frontend.review_view import render_reviews

st.set_page_config(page_title="Evidence workspace", page_icon="📖", layout="wide")
# Streamlit removes unmounted widget state when changing workspaces. Keep a small
# allowlisted input memory so returning cannot associate a result with a reset claim.
INPUT_KEYS = ["mode", "scenario", "claim", "discovery_source", "discovery_claim", "discovery_top_k",
              "discovery_from", "discovery_to", "discovery_filter", "discovery_provider_filter", "discovery_pdf_only",
              "discovery_page", "discovery_inspect", "reviewer_id", "review_output", "review_confirm",
              *["review_" + name for name in ("factual_supported", "factual_total", "citation_supported", "citation_total",
                 "citation_complete", "citation_required", "verdict_consistent", "scope_preserved",
                 "correction_supported", "correction_total", "usefulness", "notes")]]
memory = st.session_state.setdefault("input_memory", {})
for key in INPUT_KEYS:
    if key in st.session_state:
        memory[key] = st.session_state[key]
    elif key in memory:
        st.session_state[key] = memory[key]
with st.sidebar:
    st.title("Research evidence")
    active = any(key in st.session_state for key in ("discovery_job", "pipeline_job", "aggregation_job", "llm_job"))
    workspace = st.radio("Workspace", ["Find papers", "Verify evidence", "Evaluate explanations"], key="workspace", disabled=active)
    st.divider()
if workspace == "Find papers":
    render_discovery()
elif workspace == "Verify evidence":
    render_verification()
else:
    render_reviews()
for key in INPUT_KEYS:
    if key in st.session_state:
        memory[key] = st.session_state[key]
