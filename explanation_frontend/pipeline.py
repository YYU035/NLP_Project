"""Built-in local discovery → retrieval → RRF → abstract selection → NLI bridge."""
from functools import lru_cache
import json
import threading
from time import perf_counter

from .discovery import DiscoverySnapshot, LiveDiscoveryBridge
from .models import Paper, RunSnapshot, AggregationResponse
from .person4 import Person4Adapter
from .passage_selection import select_abstract
from .ranking import connect_ranking, ranking_request
from .pipeline_setup import NLI_ID, NLI_PATH, NLI_REVISION, ROOT

_MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def local_encoder():
    from sentence_transformers import SentenceTransformer
    path = ROOT / ".models" / "all-MiniLM-L6-v2"
    if not (path / "config.json").is_file():
        raise ValueError("Run setup-discovery.ps1 for local embeddings")
    return SentenceTransformer(str(path), local_files_only=True, trust_remote_code=False, device="cpu")


@lru_cache(maxsize=1)
def local_verifier():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from ._upstream.claim_verifier.verifier import ClaimVerifier
    marker = NLI_PATH / "pinned_revision.txt"
    if not marker.is_file() or marker.read_text().strip() != NLI_REVISION:
        raise ValueError("Run python -m explanation_frontend.pipeline_setup first")
    torch.set_num_threads(min(6, torch.get_num_threads()))
    # Explicit offline construction; retain the owner's classify/verify methods.
    verifier = object.__new__(ClaimVerifier)
    verifier.checkpoint, verifier.revision = NLI_ID, NLI_REVISION
    verifier.max_length, verifier.threshold, verifier.seed, verifier.device = 512, 0.50, 42, "cpu"
    verifier.tokenizer = AutoTokenizer.from_pretrained(str(NLI_PATH), local_files_only=True, trust_remote_code=False)
    verifier.model = AutoModelForSequenceClassification.from_pretrained(
        str(NLI_PATH), local_files_only=True, trust_remote_code=False, dtype=torch.float32).eval()
    verifier.label_mapping = verifier._build_label_mapping()
    if set(x.value for x in verifier.label_mapping.values()) != {"SUPPORT", "CONTRADICT", "NEUTRAL"}:
        raise ValueError("Unexpected NLI label mapping")
    return verifier


def safe_aggregate(verifier, response, subset_of_paper_ids=None):
    """Owner algorithm plus explicit conflict-side preservation (frontend addition)."""
    from ._upstream.claim_verifier.schemas import AggregationResponse
    ids = set(subset_of_paper_ids) if subset_of_paper_ids is not None else {
        p.paper_id for p in response.per_paper_verdicts}
    result = verifier.aggregate(response, subset_of_paper_ids=sorted(ids))
    papers = [p for p in response.per_paper_verdicts if p.paper_id in ids]
    conflict = any(p.conflict_flag for p in papers) or (
        any(p.verdict.value == "SUPPORTED" for p in papers) and
        any(p.verdict.value == "CONTRADICTED" for p in papers))
    if not conflict:
        return result
    passages = {p.passage_id: p for p in response.evidence_passages}
    labels = {p.passage_id: p for p in response.per_passage_labels}
    sides = []
    for field, label in [("supporting_passage_ids", "SUPPORT"), ("contradicting_passage_ids", "CONTRADICT")]:
        found = []
        for paper in papers:
            for pid in getattr(paper, field):
                verdict = labels.get(pid)
                if (pid in passages and verdict and verdict.label.value == label and
                        not verdict.truncated and verdict.operational_status.value == "OK" and
                        pid not in found):
                    found.append(pid)
        sides.append(found)
    if not all(sides):
        raise ValueError("Conflict requires eligible evidence from both sides")
    ordered = list(dict.fromkeys([sides[0][0], sides[1][0]] + sides[0][1:] + sides[1][1:]))[:5]
    return AggregationResponse(claim=response.claim, aggregated_verdict="CONFLICTING",
        primary_evidence=[passages[pid] for pid in ordered],
        global_confidence=max(p.confidence for p in papers),
        verdict_delta_reason="Frontend integration policy: preserve both eligible conflict sides, including internal paper conflict.")


class LocalPipeline:
    def __init__(self, *, encoder=None, verifier=None):
        self.encoder, self.verifier = encoder, verifier
        self.adapter = None
        self._snapshot = None

    def run(self, claim, options):
        bridge = LiveDiscoveryBridge()
        try:
            source = bridge.retrieve(claim, top_k=10)
        finally:
            bridge.close()
        return self.run_from_discovery(source, [p.identity for p in source.result.papers])

    def run_from_discovery(self, source, paper_ids):
        from ._upstream.retrieval.schema import Paper as RetrievalPaper
        from ._upstream.retrieval.retriever import Retriever
        from ._upstream.retrieval.bm25 import BM25Retriever
        from ._upstream.retrieval.sbert import SBERTRetriever
        from .models import digest
        selected = set(paper_ids)
        if len(selected) != len(paper_ids) or not selected <= {p.identity for p in source.result.papers}:
            raise ValueError("Unknown or duplicate candidate selection")
        entries = [p for p in source.result.papers if p.identity in selected]
        claim = source.result.original_claim
        start = perf_counter()
        # Preserve the original abstract as one field for retrieval; no fake sentence indices.
        records = [RetrievalPaper(paper_id=p.identity, title=p.paper.title or "",
                   abstract=[p.paper.abstract] if p.paper.abstract else [], source=p.paper.source)
                   for p in entries if (p.paper.title or "").strip() or (p.paper.abstract or "").strip()]
        ranking = source.ranking_context
        branch_scores = {}
        if ranking is None:
            request = ranking_request(source)
            if records:
                encoder = self.encoder or local_encoder()
                for name, engine in [("bm25", BM25Retriever()),
                                     ("sbert", SBERTRetriever(cache_dir=None, encoder=encoder))]:
                    retriever = Retriever(name, **{name: engine}, corpus="live").index(records, use_cache=False)
                    output = retriever.search(claim, k=len(records), claim_id=request["claim_id"])
                    for row in output.results:
                        branch_scores.setdefault(row.paper_id, {})[name] = row.score
                    request[name + "_ranking"] = [row.paper_id for row in output.results]
                    request[name + "_seconds"] = output.latency_ms / 1000
            ranking = connect_ranking(json.dumps(request), source.result)
        ordered = [r["entry"]["identity"] for r in ranking["papers"] if r["entry"]["identity"] in selected]
        # Keep no-text candidates visible as failed inputs rather than silently dropping them.
        ordered += sorted(selected - set(ordered))
        entry_map = {p.identity: p for p in entries}
        passages = {pid: select_abstract(claim, pid, entry_map[pid].paper.abstract) for pid in ordered}
        metadata = [Paper(paper_id=p.identity, title=p.paper.title or "Title unavailable",
                    authors=p.paper.authors, year=p.paper.year, url=p.paper.landing_url,
                    source_version=source.source_reference, source_status="unverified",
                    retrieval_scores={**branch_scores.get(p.identity, {}),
                        **({"rrf": next(r["rrf_score"] for r in ranking["papers"] if r["entry"]["identity"] == p.identity)}
                           if any(r["entry"]["identity"] == p.identity for r in ranking["papers"]) else {})}) for p in entries]
        if any(passages.values()):
            with _MODEL_LOCK:
                verifier = self.verifier or local_verifier()
                native = verifier.verify(claim, [{"paper_id": pid, "title": entry_map[pid].paper.title or "",
                    "evidence_type": "abstract"} for pid in ordered], passages)
            aggregate_fn = lambda response, subset_of_paper_ids: safe_aggregate(verifier, response, subset_of_paper_ids)
        else:
            # Actual owner handles empty passage inputs without neural inference.
            from ._upstream.claim_verifier.verifier import ClaimVerifier
            verifier = object.__new__(ClaimVerifier)
            verifier.checkpoint, verifier.revision = NLI_ID, NLI_REVISION
            verifier.max_length, verifier.threshold, verifier.seed = 512, .50, 42
            native = verifier.verify(claim, [{"paper_id": pid, "title": entry_map[pid].paper.title or ""} for pid in ordered], passages)
            aggregate_fn = verifier.aggregate
        adapter = Person4Adapter(aggregate_fn)
        snapshot = adapter.snapshot(native, metadata, mode=source.mode)
        context = source.model_dump(mode="json")
        # RankingInput retains the full source, while selection is exactly the user's scope.
        context["selected_paper_ids"] = sorted(selected)
        # If some selected no-text candidates were unranked, keep ranking in run diagnostics;
        # DiscoverySnapshot requires selected IDs to be a subset of ranked output.
        eligible = {r["entry"]["identity"] for r in ranking["papers"]}
        if selected <= eligible:
            context["ranking_context"] = ranking
        data = snapshot.model_dump(mode="json")
        data.update(discovery_context=context, search_complete=source.result.status in {"ok", "empty"},
            retrieved_at=(source.retrieved_at or source.imported_at).isoformat(),
            stage_statuses={"discovery": source.result.status,
                "retrieval": "supplied_ranking_inputs" if source.ranking_context else "local_bm25_sbert",
                "ranking": "upstream_rrf", "passages": "frontend_abstract_windows_v1",
                "verification": "local_nli" if any(passages.values()) else "no_passages",
                "integration_seconds": str(perf_counter() - start),
                "ranking_output_sha256": digest(ranking)},
            filters={"scope": "selected discovery candidates; abstract only", "nli_license": "MIT",
                     "retrieval_timestamp": "supplied" if source.retrieved_at else "unknown; import time used for snapshot bookkeeping",
                     "retrieval_commit": "2bc7d81292313498fd70dd13656024d00cf3dda4",
                     "ranking_commit": "a91e257ab260ded8df8c2cc2f214c4c7fd16af6c",
                     "verifier_commit": "1ac06b675e3d0719a63cb8efe98eda4f2c5fd576",
                     "ranking_source": "Yu Yue", "retrieval_source": "Naila",
                     "ranking_output_json": json.dumps(ranking, ensure_ascii=False),
                     "passage_selector": "exact abstract windows; offsets encoded in passage IDs"})
        if source.result.status == "failed":
            data["provisional"] = AggregationResponse(claim=claim, aggregated_verdict=None,
                status="failed", sufficiency_cause="DISCOVERY_FAILED").model_dump(mode="json")
            data["verification"]["processing_failures"].append(
                {"stage": "discovery", "code": "ALL_PROVIDERS_FAILED", "paper_id": None})
        elif source.result.status == "partial":
            if data["provisional"]["status"] != "failed":
                data["provisional"]["status"] = "partial"
            data["verification"]["processing_failures"].append(
                {"stage": "discovery", "code": "PARTIAL_DISCOVERY", "paper_id": None})
        elif not selected:
            data["provisional"]["sufficiency_cause"] = "NO_RETRIEVAL"
        result = RunSnapshot.model_validate(data)
        # Snapshot was enriched with provenance after mapping; tie cache to that exact result.
        adapter._fingerprint = result.fingerprint
        self.adapter, self._snapshot = adapter, result
        return result

    def aggregate(self, snapshot, paper_ids):
        if self.adapter is None:
            raise ValueError("No original response is cached")
        if snapshot.stage_statuses.get("discovery") == "failed":
            if self._snapshot is None or snapshot.fingerprint != self._snapshot.fingerprint:
                raise ValueError("No matching original response")
            if paper_ids:
                raise ValueError("Unknown selected paper")
            return snapshot.provisional
        return self.adapter.aggregate(snapshot, paper_ids)


def create_provider():
    if not (NLI_PATH / "pinned_revision.txt").is_file() or not (
            ROOT / ".models" / "all-MiniLM-L6-v2" / "config.json").is_file():
        raise ValueError("Run setup-pipeline.ps1 before using local verification")
    return LocalPipeline()
