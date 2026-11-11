"""Blinded raw-output review and explicit human score collection."""
from research_verifier.runtime.paths import EXAMPLES_DIR, EVALUATION_DIR
import json
from pathlib import Path

import streamlit as st

from research_verifier.explanations.review_study import run_study
from research_verifier.explanations.reviews import ReviewScore, load_review_packets, export_reviews, summarize_saved_reviews


def render_reviews():
    ss = st.session_state
    ss.setdefault("human_reviews", {})
    ss.setdefault("reviewer_id", "")

    def install(packets, label):
        ss.review_packets = packets
        ss.review_dataset = label
        ss.human_reviews = {}
        ss.pop("review_output", None)
        ss.pop("review_last_output", None)

    with st.sidebar:
        st.caption("Use evaluation/run.py to create blinded packets from frozen cases. Upload review_packets.jsonl, not the private method key.")
        actual=EXAMPLES_DIR/'replays/review_packets.jsonl'
        if actual.is_file() and st.button('Load actual-model review set'):
            install(load_review_packets(actual.read_bytes()),'Actual SciFact fixed-candidate runs; grounded templates, single method. Human scores pending; model verdict is a reference, not scientific truth.')
        if actual.is_file():
            st.download_button('Download blank review CSV',actual.with_name('review_blank.csv').read_bytes(),'review_blank.csv','text/csv')
        if st.button("Load synthetic review set"):
            cases = [json.loads(line) for line in (EVALUATION_DIR / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            result = run_study(cases, mode="fixture")
            packets = load_review_packets("\n".join(json.dumps(p) for p in result["review_packets"]))
            install(packets, "Synthetic harness — scripted outputs, not actual model performance")
        uploaded = st.file_uploader("Blinded review packets (JSONL)", type=["jsonl"], key="review_upload")
        if uploaded and st.button("Load review packets"):
            try:
                install(load_review_packets(uploaded.getvalue()), "Uploaded packets — dataset provenance must be recorded by the study owner")
            except Exception:
                st.error("Packets rejected: check the JSONL schema, unique IDs and 10 MB limit.")
        st.text_input("Reviewer ID", key="reviewer_id", help="Use a study alias with letters, numbers, hyphens or underscores; no personal details needed.")

    st.title("Evaluate explanation quality")
    st.write("Compare each raw explanation with its supplied evidence. Citation existence and exact quotation do not independently establish scientific support.")
    st.caption("Human scores are saved only when you submit them. Missing reviews remain pending; no score is inferred from a validator or a fallback.")
    if "review_packets" not in ss:
        st.info("Load the synthetic review set to practise, or upload blinded real-study review packets. No model calls are made in this review screen.")
        return
    packets = {p.output_id: p for p in ss.review_packets}
    st.text(ss.review_dataset)
    columns = st.columns(3)
    columns[0].metric("Outputs to review", len(packets))
    columns[1].metric("Saved human reviews", len(ss.human_reviews))
    completed = sum((pid, ss.reviewer_id) in ss.human_reviews for pid in packets)
    columns[2].metric("Pending for this reviewer", len(packets) - completed)
    current = st.selectbox("Blinded output", list(packets), key="review_output")
    packet = packets[current]
    identity = (packet.fingerprint, ss.reviewer_id)
    fields = ["factual_supported", "factual_total", "citation_supported", "citation_total", "citation_complete", "citation_required",
              "verdict_consistent", "scope_preserved", "correction_supported", "correction_total", "usefulness", "notes"]
    if ss.get("review_last_output") != identity:
        saved = ss.human_reviews.get((current, ss.reviewer_id))
        data = saved.model_dump() if saved else {}
        for field in fields:
            ss[f"review_{field}"] = data.get(field, None if field == "usefulness" else "" if field == "notes" else 0)
        ss.review_confirm = False
        ss.review_last_output = identity
    left, right = st.columns(2)
    with left:
        st.subheader("Original claim and evidence")
        st.text(packet.claim)
        st.text("Authoritative verdict: " + (packet.authoritative_verdict or "processing failure; no scientific verdict"))
        for index, passage in enumerate(packet.evidence, 1):
            with st.expander(f"Supplied passage {index}", expanded=True):
                st.text(passage.text)
                st.text(f"ID: {passage.passage_id} | paper: {passage.paper_id} | section: {passage.section} | page: {passage.page or 'unavailable'}")
        if not packet.evidence:
            st.info("No eligible decisive evidence supplied. Assess the failure/uncertainty wording rather than inventing support.")
    with right:
        st.subheader("Raw explanation under review")
        st.code(packet.raw_output, language="json", wrap_lines=True)
        st.caption("This is untrusted raw output for evaluation, including rejected generations. It is not the normal user-facing explanation.")
    with st.expander("Scoring rubric"):
        st.write("Count atomic scientific statements and statement–citation pairs. Partly supported sentences count as unsupported. Preserve numbers, units, population/benchmark scope, negation, qualifications and uncertainty.")
        st.write("Usefulness: 1 misleading/unusable · 2 major omissions · 3 understandable but limited · 4 clear and traceable · 5 clear, complete and useful for inspecting evidence. Zero denominator means not applicable, not perfect support.")
    with st.form("human_score"):
        st.subheader("Your assessment")
        a, b = st.columns(2)
        groups = [("Factual statements supported", "factual_supported"), ("Total factual statements", "factual_total"),
                  ("Supported citation pairs", "citation_supported"), ("Total citation pairs", "citation_total"),
                  ("Statements with complete citations", "citation_complete"), ("Statements requiring citations", "citation_required"),
                  ("Supported corrections", "correction_supported"), ("Total corrections", "correction_total")]
        for index, (label, field) in enumerate(groups):
            with a if index % 2 == 0 else b:
                st.number_input(label, min_value=0, max_value=10000, step=1, key=f"review_{field}")
        st.selectbox("Verdict consistent (0 no / 1 yes)", [0, 1], key="review_verdict_consistent")
        st.selectbox("Numbers, qualifications and uncertainty preserved (0 no / 1 yes)", [0, 1], key="review_scope_preserved")
        st.selectbox("Usefulness (1–5)", [None, 1, 2, 3, 4, 5], key="review_usefulness",
                     format_func=lambda value: "Choose a rating" if value is None else str(value))
        st.text_area("Review notes / failure analysis", key="review_notes")
        st.checkbox("I reviewed the raw output against the supplied evidence", key="review_confirm")
        saved = st.form_submit_button("Save human review", type="primary")
    if saved:
        try:
            if not ss.review_confirm:
                raise ValueError("Confirmation missing")
            record = ReviewScore(output_id=current, reviewer=ss.reviewer_id,
                                 **{field: ss[f"review_{field}"] for field in fields})
            ss.human_reviews[(current, ss.reviewer_id)] = record
            ss.review_saved_notice = current
            st.rerun()
        except ValueError:
            st.error("Review not saved. Provide a valid reviewer ID, explicit usefulness rating, confirmation, and supported/complete counts no greater than their totals.")
    if ss.get("review_saved_notice") == current:
        st.success("Saved your human review. Submitting again updates this output/reviewer pair.")
    records = list(ss.human_reviews.values())
    st.download_button("Download human reviews CSV", export_reviews(records), "human-reviews.csv", "text/csv", disabled=not records)
    st.download_button("Download human reviews JSON", json.dumps([r.model_dump() for r in records], indent=2), "human-reviews.json", "application/json", disabled=not records)
    with st.expander("Summary of submitted human scores"):
        st.json(summarize_saved_reviews(records))
    st.caption("Download before closing the session; reviews are not written to a shared server file. Join the private method key with evaluation/run.py after independent scoring. Do not treat practice scores as final research results.")
