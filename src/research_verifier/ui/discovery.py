"""Compact discovery workspace, independent of downstream scientific verification."""
import json
import math

import streamlit as st

from research_verifier.discovery.service import LiveDiscoveryBridge, import_discovery, export_discovery, sample_discovery, discovery_handoff, filter_candidates
from research_verifier.runtime.session import start_job, poll_job
from research_verifier.explanations.grounding import safe_url
from research_verifier.retrieval.ranking import connect_ranking, ranking_request, ranked_candidates
from research_verifier.discovery.abstract_quality import quality_label, review_discovery
from research_verifier.ui.abstract_quality import render_abstract_quality, render_quality_summary


def render_discovery():
    ss = st.session_state
    ss.setdefault("discovery_source", "Recorded example")
    ss.setdefault("discovery_claim", "")
    ss.setdefault("discovery_retired", [])

    def clear_result():
        if "discovery_job" in ss:
            ss.discovery_retired.append(ss.pop("discovery_job"))
        ss.pop("discovery_snapshot", None)
        ss.pop("discovery_error", None)

    def install(snapshot):
        from research_verifier.runtime.session import install_discovery_state
        install_discovery_state(ss,snapshot)

    def reset_page():
        ss.discovery_page = 1

    def select_paper(pid, key):
        ss.discovery_selection[pid] = ss[key]

    def prepare_verification():
        selected = [pid for pid, included in ss.discovery_selection.items() if included]
        ss.pending_discovery = {"snapshot": ss.discovery_snapshot, "ids": selected}
        for name in ("snapshot", "provider", "generation", "replay", "operation_error"):
            ss.pop(name, None)
        ss.mode = "Selected search"
        ss.claim = ss.discovery_snapshot.result.original_claim
        ss.workspace = "Verify evidence"

    ss.discovery_retired = [j for j in ss.discovery_retired if not j.future.done()]
    busy = "discovery_job" in ss or bool(ss.discovery_retired)
    with st.sidebar:
        st.selectbox("Search source", ["Recorded example", "Upload search", "Live search"],
                     key="discovery_source", on_change=clear_result)
        if ss.discovery_source == "Recorded example":
            st.caption("A saved real search from Raghavi's branch; no new API calls or scientific labels.")
            if st.button("Load recorded search", type="primary", disabled=busy):
                clear_result()
                install(sample_discovery())
        elif ss.discovery_source == "Upload search":
            uploaded = st.file_uploader("Person 1 RetrievalResult or discovery export", type=["json"], key="discovery_upload")
            if uploaded and st.button("Load search file", disabled=busy):
                try:
                    snapshot = import_discovery(uploaded.getvalue())
                    clear_result()
                    install(snapshot)
                except Exception:
                    ss.discovery_error = "Search file rejected: check its size, schema, claim, identity/provenance links and checksum."
        else:
            st.info("Live search requires the bundled discovery package and local models. Run setup.cmd first; no models download during search.")
            st.number_input("Candidates per provider", min_value=1, max_value=30, value=30,
                            key="discovery_top_k", on_change=clear_result)
            st.date_input("Published from (optional)", value=None, key="discovery_from", on_change=clear_result)
            st.date_input("Published to (optional)", value=None, key="discovery_to", on_change=clear_result)
        if "discovery_bridge" in ss and not busy:
            if st.button("Release live search connection"):
                ss.pop("discovery_bridge").close()
        st.caption("A PDF URL is a metadata link, not proof that full text was inspected. Discovery order is not a scientific relevance ranking.")

    st.title("Find papers for your claim")
    st.write("Discover candidate papers, inspect their sources, and prepare a traceable handoff for evidence verification.")

    @st.fragment(run_every=0.5 if busy else None)
    def poll():
        if ss.discovery_retired and all(j.future.done() for j in ss.discovery_retired):
            ss.discovery_retired = []
            st.rerun()
        if "discovery_job" not in ss:
            return
        job = ss.discovery_job
        status, result = poll_job(job)
        if status == "pending":
            st.info("Searching providers and preserving query provenance…")
            return
        ss.pop("discovery_job")
        if status == "done" and result.result.original_claim == ss.discovery_claim:
            # This poll runs before claim widgets, so exact claim installation is safe.
            install(result)
        else:
            ss.discovery_retired.append(job)
            ss.discovery_error = "Live search failed or timed out. Check the package, local model and provider setup; no demo result was substituted."
        st.rerun()

    poll()
    if ss.discovery_source != "Live search" and "discovery_snapshot" in ss:
        # Cached claims are authoritative data, not editable widget defaults.
        ss.discovery_claim = ss.discovery_snapshot.result.original_claim
    st.text_area("Original scientific claim", key="discovery_claim", max_chars=20_000,
                 disabled=ss.discovery_source != "Live search", on_change=clear_result,
                 help="Search reformulations never replace this claim for evidence assessment.")
    if ss.discovery_source == "Live search" and st.button("Find papers", type="primary", disabled=busy):
        clear_result()
        claim = ss.discovery_claim
        if not claim.strip():
            ss.discovery_error = "Enter a nonempty scientific claim."
        elif ss.discovery_from and ss.discovery_to and ss.discovery_from > ss.discovery_to:
            ss.discovery_error = "The publication start date must not follow the end date."
        else:
            ss.setdefault("discovery_bridge", None)
            if ss.discovery_bridge is None:
                ss.discovery_bridge = LiveDiscoveryBridge()
            bridge = ss.discovery_bridge
            options = dict(top_k=ss.discovery_top_k, from_publication_date=ss.discovery_from,
                           to_publication_date=ss.discovery_to)
            ss.discovery_job = start_job(lambda c=claim, b=bridge, o=options: b.retrieve(c, **o), claim, timeout=120)
            st.rerun()
    if "discovery_job" in ss and st.button("Cancel search"):
        clear_result()
        st.rerun()
    if ss.discovery_retired:
        st.warning("An abandoned search is finishing; its result is discarded and new searches wait for it.")
    if "discovery_error" in ss:
        st.error(ss.discovery_error)
    if "discovery_snapshot" not in ss:
        st.info("Load the recorded example, import a saved search, or configure live discovery. Verification remains a separate step.")
        return

    snapshot = ss.discovery_snapshot
    result = snapshot.result
    quality_review = review_discovery(snapshot)
    quality_by_identity = quality_review.by_identity
    with st.expander("Connect BM25/SBERT rankings and Person 3 RRF"):
        st.write("Download the input contract for Person 2 to fill with ordered canonical IDs. Upload their RankingInput to run Yu Yue's pinned RRF locally, or upload an RRF-only RankingOutput. No API or model is called.")
        st.caption("The downloaded branch lists are empty placeholders. Ranking scores measure retrieval relevance, not truth. Cross-encoder integration remains pending.")
        st.download_button("Download Person 2 ranking input", json.dumps(ranking_request(snapshot), indent=2),
                           "ranking-input-unfilled.json", "application/json")
        upload = st.file_uploader("RankingInput or RRF RankingOutput for this exact search", type=["json"],
                                  key=f"ranking-upload:{snapshot.result.original_claim}")
        if st.button("Connect ranking", disabled=upload is None or busy):
            try:
                ranking = connect_ranking(upload.getvalue(), result)
                updated = snapshot.model_dump(mode="json")
                updated["ranking_context"] = ranking
                updated["selected_paper_ids"] = [r["entry"]["identity"] for r in ranking["papers"]]
                install(type(snapshot).model_validate(updated))
                st.rerun()
            except Exception:
                st.error("Ranking rejected. Check the exact search payload, unique canonical IDs, RRF rows and configuration. Install discovery dependencies if paper_search is unavailable. Cross-encoder outputs are not enabled.")
        if snapshot.ranking_context is not None:
            st.download_button("Download connected ranking output", json.dumps(snapshot.ranking_context, indent=2),
                               "ranking-output.json", "application/json")
            if st.button("Restore discovery order", disabled=busy):
                install(snapshot.model_copy(update={"ranking_context": None,
                    "selected_paper_ids": [p.identity for p in result.papers]}))
                st.rerun()
    if snapshot.ranking_context is not None:
        ranking = snapshot.ranking_context
        st.info(f"RRF ranking connected: {len(ranking['papers'])} displayed papers. Papers outside the ranked output are excluded from selection. No scientific verification has run.")
        st.caption(f"Ranking stage only: {ranking['ranking_seconds']:.4f}s; excludes discovery, retrieval and model loading.")
        for warning in ranking["warnings"]:
            st.text(warning)
    st.caption("LIVE DISCOVERY" if snapshot.mode == "live" else "CACHED DISCOVERY · not a current search")
    st.text("Source: " + snapshot.source_reference)
    st.text("Retrieval time: " + (snapshot.retrieved_at.isoformat() if snapshot.retrieved_at else "not supplied by the native saved payload"))
    st.text(f"Publication filter: {result.from_publication_date or 'none'} to {result.to_publication_date or 'none'} | limit: {result.top_k} per provider")
    if result.status == "failed":
        st.error("All discovery providers failed. No scientific verdict was produced.")
    elif result.status == "partial":
        st.warning("Search is partial: some providers or queries failed. Available candidates are retained.")
    elif result.status == "empty":
        st.info("Search completed with zero candidates. This is not a scientific claim verdict.")
    selected = [pid for pid, included in ss.discovery_selection.items() if included]
    summary = st.columns(4)
    summary[0].metric("Candidates", len(result.papers))
    summary[1].metric("Selected", len(selected))
    summary[2].metric("Search seconds", f"{result.seconds:.2f}")
    summary[3].metric("HTTP attempts", result.http_attempts)
    status_cols = st.columns(len(result.providers))
    for column, (name, provider) in zip(status_cols, result.providers.items()):
        with column:
            count = sum(any(o.provider == name for o in p.provenance) for p in result.papers)
            st.text(f"{name}: {provider.status}")
            st.caption(f"{count} candidate origins · {provider.seconds:.2f}s")
    with st.expander("Search queries and provider diagnostics"):
        for name, provider in result.providers.items():
            st.text(f"{name}: {provider.status}; elapsed {provider.seconds:.2f}s")
            for query in provider.queries:
                st.text(f"{query.id} | {query.strategy} | {query.stage} | {query.status} | parsed: {query.returned}")
                st.text(query.query)
                for note in query.notes:
                    st.text(note)
                for attempt in query.attempts:
                    st.text(f"HTTP {attempt.status} · {attempt.seconds:.2f}s")
            if provider.error:
                st.caption("Provider error recorded. Raw exception text is withheld from this view.")
    render_quality_summary(quality_review, key_prefix="discovery")
    st.divider()
    st.text_input("Filter by title, abstract, author or identity", key="discovery_filter", on_change=reset_page)
    st.multiselect("Provider origins (empty means all)", list(result.providers), key="discovery_provider_filter", on_change=reset_page)
    st.toggle("PDF metadata link available", key="discovery_pdf_only", on_change=reset_page)
    candidates = filter_candidates(result, text=ss.discovery_filter, providers=ss.discovery_provider_filter,
                                   pdf_only=ss.discovery_pdf_only)
    candidates = ranked_candidates(candidates, snapshot.ranking_context)
    st.caption(f"Showing {len(candidates)} matching candidates. Filtering does not change the scientific claim or rerank evidence.")
    if not candidates:
        st.info("No papers match this view. Existing selections are retained.")
    else:
        pages = list(range(1, math.ceil(len(candidates) / 10) + 1))
        if ss.get("discovery_page", 1) not in pages:
            ss.discovery_page = 1
        st.selectbox("Page (10 papers)", pages, key="discovery_page")
        left, right = st.columns([1, 1.35])
        with left:
            st.subheader("Candidate papers")
            for offset, entry in enumerate(candidates[(ss.discovery_page - 1) * 10:ss.discovery_page * 10], 1):
                pid = entry.identity
                key = f"discovery-paper:{snapshot.fingerprint}:{pid}"
                ss[key] = ss.discovery_selection[pid]
                with st.container(border=True):
                    st.checkbox(f"Include candidate {(ss.discovery_page - 1) * 10 + offset}", key=key,
                                on_change=select_paper, args=(pid, key))
                    st.text(entry.paper.title or "Title unavailable")
                    st.caption("Abstract check: " + quality_label(quality_by_identity[pid]))
                    if snapshot.ranking_context is not None:
                        row = next(r for r in snapshot.ranking_context["papers"] if r["entry"]["identity"] == pid)
                        st.text(f"RRF rank {row['rank']} · score {row['rrf_score']:.6f} · BM25 rank {row['bm25_rank']} · SBERT rank {row['sbert_rank']}")
                    st.text(f"{entry.paper.year or 'Year unavailable'} · {', '.join(sorted({o.provider for o in entry.provenance}))}")
                    st.text("Not yet verified")
        with right:
            st.subheader("Inspect a paper")
            entries = {p.identity: p for p in candidates}
            if ss.get("discovery_inspect") not in entries:
                ss.discovery_inspect = next(iter(entries))
            pid = st.selectbox("Paper to inspect", list(entries), key="discovery_inspect",
                               format_func=lambda key: (entries[key].paper.title or "Title unavailable")[:95])
            entry = entries[pid]
            paper = entry.paper
            st.text(paper.title or "Title unavailable")
            st.text("Authors: " + (", ".join(paper.authors) or "unavailable"))
            st.text(f"Published: {paper.publication_date or paper.year or 'unavailable'} | DOI: {paper.doi or 'unavailable'}")
            st.text("Identity: " + entry.identity)
            st.text("Identity aliases: " + ", ".join(entry.identity_aliases))
            st.write("Abstract / provider description")
            st.text(paper.abstract or "No abstract or description was supplied.")
            render_abstract_quality(quality_by_identity[pid])
            st.caption("Provider descriptions, especially OpenAIRE records, are not guaranteed scientific abstracts or verification evidence.")
            if safe_url(paper.landing_url):
                st.link_button("Open source record", safe_url(paper.landing_url))
            pdfs = list(dict.fromkeys(([paper.pdf_url] if paper.pdf_url else []) + paper.pdf_urls))
            for number, url in enumerate(pdfs[:3], 1):
                if safe_url(url):
                    st.link_button(f"PDF metadata link {number} (unverified)", safe_url(url))
            st.write("All retrieval origins")
            for origin in entry.provenance:
                st.text(f"{origin.provider} | query {origin.query_id} | provider rank {origin.rank} | record {origin.source_id}")
            with st.expander("Original normalized source records"):
                st.json([record.model_dump(mode="json") for record in entry.source_records])
    st.divider()
    st.subheader("Continue to evidence verification")
    st.info("These papers have not been checked for evidence or assigned scientific verdicts. The selected handoff retains any connected ranking and original provenance. The real evidence/verification adapter from Persons 3–4 remains required.")
    st.button("Prepare selected papers for verification", disabled=not selected or busy, on_click=prepare_verification)
    st.download_button("Download search snapshot", export_discovery(snapshot, selected), "discovery-result.json", "application/json")
    st.download_button("Download selected-paper handoff", json.dumps(discovery_handoff(snapshot, selected), indent=2, ensure_ascii=False),
                       "selected-papers.json", "application/json", disabled=not selected)
