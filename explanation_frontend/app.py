"""Run: python -m streamlit run explanation_frontend/app.py"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

# Supports both the repository command and a module-only checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from explanation_frontend.export import export_snapshot, import_snapshot
from explanation_frontend.llm import OpenAIClient, generate_llm
from explanation_frontend.models import make_context, digest
from explanation_frontend.providers import FixtureProvider, fixture_names, replay_aggregate
from explanation_frontend.state import Workspace, start_job, poll_job
from explanation_frontend.templates import build_template
from explanation_frontend.validation import safe_url, validate_explanation

st.set_page_config(page_title="Evidence workspace", page_icon="📖", layout="wide")
ss = st.session_state
ss.setdefault("workspace", Workspace())
ss.setdefault("mode", "Demo")
ss.setdefault("scenario", "conflict")
ss.setdefault("claim", FixtureProvider().run_fixture("conflict").claim)
ss.setdefault("retired_jobs", [])
ss.setdefault("aggregation_cache", {})
ss.setdefault("aggregation_errors", set())


def invalidate():
    ss.workspace.invalidate()
    for name in ("pipeline_job", "llm_job", "aggregation_job"):
        if name in ss:
            ss.retired_jobs.append(ss.pop(name))
    for name in ("snapshot", "provider", "generation", "replay", "operation_error"):
        ss.pop(name, None)
    ss.aggregation_cache, ss.aggregation_errors = {}, set()


def change_scenario():
    invalidate()
    ss.claim = FixtureProvider().run_fixture(ss.scenario).claim


def change_mode():
    invalidate()
    if ss.mode == "Demo":
        ss.claim = FixtureProvider().run_fixture(ss.scenario).claim


def abandon_generation():
    for name in ("llm_job", "aggregation_job"):
        if name in ss:
            ss.retired_jobs.append(ss.pop(name))
    ss.pop("generation", None)


def update_selection(pid, widget_key):
    # Keep domain state separate from widget state: fragment/full reruns may clean widgets.
    ss.paper_selection[pid] = ss[widget_key]
    abandon_generation()


def load_live_provider():
    # This is trusted server configuration, never a path or code supplied by a paper/upload.
    spec = os.environ.get("EXPLANATION_PIPELINE_FACTORY", "")
    if not spec or ":" not in spec:
        raise ValueError("Configure the trusted upstream pipeline factory first")
    module, name = spec.split(":", 1)
    provider = getattr(importlib.import_module(module), name)()
    if not callable(getattr(provider, "run", None)) or not callable(getattr(provider, "aggregate", None)):
        raise ValueError("Invalid pipeline provider")
    return provider


def install_snapshot(snapshot, provider):
    ss.snapshot, ss.provider = snapshot, provider
    ss.aggregation_cache, ss.aggregation_errors = {}, set()
    ss.selection_snapshot = snapshot.fingerprint
    ss.paper_selection = {pid: True for pid in snapshot.papers}
    for pid in snapshot.papers:
        ss[f"paper:{snapshot.fingerprint}:{pid}"] = True


ss.retired_jobs = [job for job in ss.retired_jobs if not job.future.done()]
busy = bool(ss.retired_jobs or any(name in ss for name in ("pipeline_job", "llm_job", "aggregation_job")))

with st.sidebar:
    st.title("Evidence workspace")
    st.caption("Research claim verification")
    st.selectbox("Evidence source", ["Demo", "Cached replay", "Live"], key="mode", on_change=change_mode)
    if ss.mode == "Demo":
        st.selectbox("Synthetic scenario", fixture_names(), key="scenario", on_change=change_scenario)
        st.info("Demo uses invented papers and fixed claims. It makes no scientific finding.")
    elif ss.mode == "Live":
        st.info("Live requires your teammates' configured pipeline. There is no automatic demo fallback.")
    else:
        upload = st.file_uploader("Replay a result JSON (up to 10 MB)", type=["json"])
        if upload and st.button("Load replay"):
            try:
                if upload.size > 10_000_000:
                    raise ValueError("File too large")
                bundle = import_snapshot(upload.getvalue())
                invalidate()
                ss.claim = bundle.snapshot.claim
                install_snapshot(bundle.snapshot, None)
                ss.replay = bundle
                for pid in bundle.snapshot.papers:
                    ss[f"paper:{bundle.snapshot.fingerprint}:{pid}"] = pid in bundle.selected_paper_ids
                    ss.paper_selection[pid] = pid in bundle.selected_paper_ids
                st.success("Validated replay loaded; no network request was made.")
            except Exception:
                st.error("Replay rejected: check its schema, checksum, claim and citation references.")
    st.divider()
    st.caption("Scores describe model output or relevance, not scientific truth probabilities.")
    st.caption("A citation count, recent date or open-access badge does not establish correctness.")

st.title("Inspect the evidence behind a claim")
st.write("Verify once, inspect each source, then compare the verdict over the papers you select.")
st.text_area("Scientific claim", key="claim", height=100, on_change=invalidate, max_chars=20_000,
             help="The exact claim is preserved. Editing it clears the previous result.")

left, right = st.columns([1, 4])
with left:
    submitted = st.button("Verify claim", type="primary", disabled=busy or ss.mode == "Cached replay")
with right:
    if ss.retired_jobs:
        st.warning("A cancelled/timed-out request is finishing. New remote requests wait for it; its result will be discarded.")
    if any(name in ss for name in ("pipeline_job", "llm_job", "aggregation_job")):
        if st.button("Cancel pending request"):
            if "pipeline_job" in ss:
                invalidate()
            else:
                abandon_generation()
            st.rerun()

if submitted:
    invalidate()
    if not ss.claim.strip():
        ss.operation_error = "Enter a nonempty scientific claim."
    else:
        token = ss.workspace.begin(ss.claim, {"mode": ss.mode, "scenario": ss.scenario})
        try:
            if ss.mode == "Demo":
                provider = FixtureProvider()
                snapshot = provider.run(ss.claim, {"scenario": ss.scenario})
                install_snapshot(snapshot, provider)
            else:
                provider = load_live_provider()
                ss.provider = provider
                ss.pipeline_job = start_job(lambda claim=ss.claim, p=provider: p.run(claim, {}), token, timeout=90)
                st.rerun()
        except Exception:
            ss.operation_error = ("Demo accepts only the scenario's exact synthetic claim. Choose a scenario again to restore it."
                                  if ss.mode == "Demo" else "Live pipeline is unavailable. Configure its factory and verify the upstream handoff.")


@st.fragment(run_every=0.5 if any(name in ss for name in ("pipeline_job", "llm_job", "aggregation_job")) or ss.retired_jobs else None)
def poll_requests():
    if ss.retired_jobs and all(j.future.done() for j in ss.retired_jobs):
        ss.retired_jobs = []
        st.rerun()
    for name in ("pipeline_job", "llm_job", "aggregation_job"):
        if name not in ss:
            continue
        job = ss[name]
        status, result = poll_job(job)
        if status == "pending":
            st.info("Processing pipeline…" if name == "pipeline_job" else "Aggregating selected evidence…" if name == "aggregation_job" else "Generating and checking explanation…")
            continue
        ss.pop(name)
        if status != "done":
            ss.retired_jobs.append(job)
            ss.operation_error = "Request timed out or failed. No new scientific verdict was inferred from this error."
            if name == "aggregation_job":
                ss.aggregation_errors.add(job.key)
        elif name == "pipeline_job":
            try:
                from explanation_frontend.models import RunSnapshot
                snapshot = RunSnapshot.model_validate(result)
                if snapshot.claim != ss.claim or snapshot.mode != "live":
                    raise ValueError("Stale or incorrect snapshot")
                if ss.workspace.accept(job.key, snapshot):
                    install_snapshot(snapshot, ss.provider)
            except Exception:
                ss.operation_error = "Upstream snapshot failed claim, mode or schema validation."
        elif name == "aggregation_job":
            ss.aggregation_cache[job.key] = result
        else:
            ss.generation = result
        st.rerun()


poll_requests()
if "operation_error" in ss:
    st.error(ss.operation_error)


def show_verdict(label, aggregate):
    st.metric(label, aggregate.aggregated_verdict or "PROCESSING FAILURE")
    if aggregate.status != "ok":
        st.warning("Processing status: " + aggregate.status)
    if aggregate.global_confidence is not None:
        st.caption(f"Verifier model score: {aggregate.global_confidence:.3f} · not a truth probability")
    if aggregate.sufficiency_cause:
        st.text("Evidence sufficiency: " + aggregate.sufficiency_cause)


def show_explanation(explanation, context):
    if not validate_explanation(explanation, context).displayable:
        st.error("Explanation withheld: grounding checks did not pass.")
        return
    st.caption("Method: " + explanation.method + " · exact evidence quotations; semantic support is not independently proven")
    if not explanation.statements:
        st.info("No eligible decisive passages are available for this selection.")
    for i, statement in enumerate(explanation.statements, 1):
        st.text(statement.text)
        st.text("Citations: " + ", ".join(statement.citation_ids))
        with st.expander(f"Inspect citation {i}"):
            for pid in statement.citation_ids:
                passage = context.passages[pid]
                st.text(context.snapshot.papers[passage.paper_id].title)
                st.text(passage.text)
                st.text(f"Passage: {pid} | section: {passage.section} | page: {passage.page or 'unavailable'} | sentences: {passage.sentence_indices or 'unavailable'}")
    if explanation.correction:
        st.write("Explicitly supported alternative wording")
        st.text(explanation.correction.text)
        st.text("Citations: " + ", ".join(explanation.correction.citation_ids))
    for limitation in explanation.limitations:
        st.text(limitation)


if "snapshot" in ss:
    snapshot = ss.snapshot
    if ss.get("selection_snapshot") != snapshot.fingerprint:
        ss.selection_snapshot = snapshot.fingerprint
        ss.paper_selection = {pid: ss.get(f"paper:{snapshot.fingerprint}:{pid}", True) for pid in snapshot.papers}
    st.divider()
    st.caption("CACHED REPLAY" if ss.mode == "Cached replay" else "SYNTHETIC DEMO" if snapshot.mode == "demo" else "LIVE SEARCH")
    st.text(f"Retrieved: {snapshot.retrieved_at.isoformat()} | original mode: {snapshot.mode} | search complete: {snapshot.search_complete}")
    st.text("Filters: " + str(snapshot.filters))
    with st.expander("Processing and model settings"):
        st.json(snapshot.stage_statuses)
        st.json(snapshot.verification.model_settings.model_dump())
        for failure in snapshot.verification.processing_failures:
            st.text(f"{failure.stage}: {failure.code} | paper: {failure.paper_id or 'not specified'}")
    st.subheader("1 · All-paper provisional result")
    show_verdict("Provisional verdict", snapshot.provisional)
    all_context = make_context(snapshot, list(snapshot.papers), snapshot.provisional)
    with st.expander("Provisional explanation"):
        show_explanation(build_template(all_context), all_context)

    st.subheader("2 · Inspect and select papers")
    if not snapshot.papers:
        st.info("No papers are available in this snapshot.")
    selected = []
    for i, (pid, paper) in enumerate(snapshot.papers.items(), 1):
        with st.container(border=True):
            widget_key = f"paper:{snapshot.fingerprint}:{pid}"
            ss[widget_key] = ss.paper_selection[pid]
            include = st.checkbox(f"Include paper {i}", key=widget_key,
                                  on_change=update_selection, args=(pid, widget_key))
            if include:
                selected.append(pid)
            st.text(paper.title)
            st.text(f"{', '.join(paper.authors) or 'Authors unavailable'} | {paper.year or 'Year unavailable'} | {paper.venue or 'Venue unavailable'}")
            st.caption("Paper provenance and availability")
            st.text(f"ID: {pid} | source status: {paper.source_status} | version: {paper.source_version or 'unknown'}")
            st.text(f"Full text available: {paper.has_fulltext} | open access: {paper.is_open_access if paper.is_open_access is not None else 'unknown'}")
            if safe_url(paper.url):
                st.link_button("Open paper source", safe_url(paper.url))
            verdict = snapshot.verification.per_paper_verdicts.get(pid)
            if verdict:
                st.text(f"Paper verdict: {verdict.verdict} | inspected evidence: {verdict.evidence_type} | conflict: {verdict.conflict_flag}")
                if verdict.confidence is not None:
                    st.text(f"Paper verifier model score: {verdict.confidence:.3f} — not a truth probability")
                if verdict.sufficiency_cause:
                    st.text("Limitation: " + verdict.sufficiency_cause)
            else:
                st.warning("No paper verdict is available; inspect processing failures.")
            for name, score in paper.retrieval_scores.items():
                st.text(f"Retrieval score ({name}): {score:.4g} — relevance only")
            for n, passage in enumerate((p for p in snapshot.verification.evidence_passages if p.paper_id == pid), 1):
                with st.expander(f"Evidence passage {n}"):
                    st.text(passage.text)
                    st.text(f"ID: {passage.passage_id} | section: {passage.section} | page: {passage.page or 'unavailable'} | version: {passage.source_version or 'unknown'}")
                    label = snapshot.verification.per_passage_labels.get(passage.passage_id)
                    st.text(f"Passage label: {label.label if label else 'unavailable'} | truncated: {label.truncated if label else 'unknown'}")
                    if label and label.confidence is not None:
                        st.text(f"Passage verifier model score: {label.confidence:.3f} — not a truth probability")
    try:
        replay = ss.get("replay")
        if replay and sorted(selected) == replay.selected_paper_ids:
            aggregation = replay.aggregation
        elif set(selected) == set(snapshot.papers):
            aggregation = snapshot.provisional
        elif ss.mode == "Cached replay":
            st.info("Changed replay selection uses the documented local replay policy, not a new verifier run.")
            aggregation = replay_aggregate(snapshot, selected)
        elif ss.mode == "Live":
            aggregation_key = digest({"snapshot": snapshot.fingerprint, "selected": sorted(selected)})
            if aggregation_key in ss.aggregation_errors:
                st.error("Subset aggregation failed or timed out. No selected verdict is available.")
                if st.button("Retry subset aggregation", disabled=busy):
                    ss.aggregation_errors.discard(aggregation_key)
                    st.rerun()
                st.stop()
            if aggregation_key not in ss.aggregation_cache:
                if not busy:
                    ss.aggregation_job = start_job(
                        lambda p=ss.provider, s=snapshot, ids=list(selected): p.aggregate(s, ids),
                        aggregation_key, timeout=5)
                    st.rerun()
                st.info("Waiting for subset aggregation. The provisional result above still covers all papers.")
                st.stop()
            aggregation = ss.aggregation_cache[aggregation_key]
        else:
            aggregation = ss.provider.aggregate(snapshot, selected)
        context = make_context(snapshot, selected, aggregation)
    except Exception:
        st.error("Subset aggregation failed. The provisional result remains tied to all papers; no selected verdict is available.")
        st.stop()

    st.subheader("3 · Verdict over your selection")
    show_verdict("Selected-evidence verdict", aggregation)
    st.text(f"Selected {len(selected)} of {len(snapshot.papers)} papers. Provisional: {snapshot.provisional.aggregated_verdict or 'processing failure'} → selected: {aggregation.aggregated_verdict or 'processing failure'}")
    st.text(aggregation.verdict_delta_reason)
    template = build_template(context)
    generated = ss.get("generation")
    if generated and generated.displayed.context_key != context.key:
        ss.pop("generation", None)
        generated = None
    explanation = generated.displayed if generated else replay.explanation if replay and sorted(selected) == replay.selected_paper_ids else template
    st.subheader("Grounded explanation")
    show_explanation(explanation, context)
    if generated:
        st.text(f"LLM status: {generated.status} | model: {generated.model} | prompt: {generated.prompt_version} | seconds: {generated.elapsed_seconds:.2f}")
        if generated.status != "accepted":
            st.warning("Template shown because generation was unavailable or failed grounding checks.")
            st.text("Check codes: " + ", ".join(generated.issues))
        if generated.omitted_passage_ids:
            st.text("Passages omitted from generation: " + ", ".join(generated.omitted_passage_ids))
    if st.button("Generate LLM explanation", disabled=busy or aggregation.status == "failed" or not aggregation.primary_evidence,
                 help="Sends the exact claim and selected evidence to the configured OpenAI model. Templates require no key."):
        try:
            client = OpenAIClient.from_env()
            ss.llm_job = start_job(lambda c=context, cl=client: generate_llm(c, cl), context.key,
                                  timeout=client.timeout * 2 + 5)
            st.rerun()
        except (ValueError, TypeError):
            st.error("Configure OPENAI_API_KEY, EXPLANATION_LLM_MODEL and a valid timeout in the server environment.")
    st.download_button("Download replayable result", export_snapshot(context, explanation),
                       file_name="evidence-result.json", mime="application/json")
    st.caption("Exports include your exact claim and evidence. Source checksums detect accidental changes, not authenticity.")
else:
    st.info("Choose a synthetic scenario and verify the fixed claim, upload a saved result, or configure the live pipeline.")
