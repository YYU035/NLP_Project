"""Frozen SciFact-dev QC, not an untouched test set or human evaluation."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

from ..pipeline import local_encoder, local_verifier, safe_aggregate
from ..person4 import Person4Adapter
from ..passage_selection import select_abstract
from ..models import Paper, make_context
from ..templates import build_template
from ..validation import validate_explanation
from ..export import export_snapshot
from .._upstream.retrieval.schema import Paper as RetrievalPaper
from .._upstream.retrieval.retriever import build_all

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1ac06b675e3d0719a63cb8efe98eda4f2c5fd576"


def data_file(name):
    raw = subprocess.check_output(["git", "show", f"{REVISION}:claim_verifier/data/{name}"], cwd=ROOT.parent)
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()], hashlib.sha256(raw).hexdigest()


def main():
    started = perf_counter()
    output = ROOT / "outputs" / "development_qc"
    output.mkdir(parents=True, exist_ok=True)
    corpus, corpus_hash = data_file("corpus.jsonl")
    claims, claims_hash = data_file("claims_dev.jsonl")
    docs = {str(p["doc_id"]): p for p in corpus}
    rows = []
    for claim in claims:
        for pid, evidence in claim.get("evidence", {}).items():
            for item in evidence:
                if item["label"] in {"SUPPORT", "CONTRADICT"} and item["sentences"]:
                    rows.append({"claim_id": str(claim["id"]), "claim": claim["claim"], "paper_id": pid,
                        "gold": item["label"], "sentence_indices": item["sentences"]})
    # Frozen label-stratified development sample. No outcome-based selection/tuning.
    key = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
    selected = sum([sorted([x for x in rows if x["gold"] == label], key=key)[:12]
                    for label in ["SUPPORT", "CONTRADICT"]], [])
    manifest = {"dataset": "SciFact-dev", "purpose": "development QC; owner already calibrated on dev",
                "selection": "12 per annotated label, SHA256 order; no neutral pseudo-labels",
                "corpus_sha256": corpus_hash, "claims_sha256": claims_hash, "cases": selected}
    (output / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Frozen 24 annotated development pairs before inference", flush=True)
    verifier = local_verifier()
    from .._upstream.claim_verifier.schemas import EvidencePassage
    records, review_cases = [], []
    for index, row in enumerate(selected):
        doc = docs[row["paper_id"]]
        oracle = " ".join(doc["abstract"][i] for i in row["sentence_indices"])
        prediction = verifier._classify_passage(row["claim"], EvidencePassage(
            passage_id=f"oracle:{index}", paper_id=row["paper_id"], section="abstract", text=oracle))
        passages = select_abstract(row["claim"], row["paper_id"], " ".join(doc["abstract"]))
        native = verifier.verify(row["claim"], [{"paper_id": row["paper_id"], "title": doc["title"]}],
                                 {row["paper_id"]: passages})
        adapter = Person4Adapter(lambda response, subset_of_paper_ids:
                                 safe_aggregate(verifier, response, subset_of_paper_ids))
        snapshot = adapter.snapshot(native, [Paper(paper_id=row["paper_id"], title=doc["title"],
            source_status="SciFact-dev oracle document", source_version=corpus_hash)])
        context = make_context(snapshot, list(snapshot.papers), snapshot.provisional)
        explanation = build_template(context)
        gate = validate_explanation(explanation, context)
        bundle = export_snapshot(context, explanation)
        name = f"case-{index:02d}.json"
        (output / name).write_text(bundle, encoding="utf-8")
        if index in [0, 1, 2, 12, 13, 14]:
            review_cases.append({"case_id": f"dev-{index:02d}", "snapshot_file": name,
                                 "dataset": "SciFact-dev-oracle-document-QC", "split": "development"})
        # Use 100 repeats to measure warm subset mapping/validation, not inference.
        t0 = perf_counter()
        for _ in range(100):
            adapter.aggregate(snapshot, list(snapshot.papers))
        records.append({**row, "oracle_prediction": prediction.label.value,
            "oracle_truncated": prediction.truncated,
            "selected_window_verdict": snapshot.provisional.aggregated_verdict,
            "displayable": gate.displayable, "subset_mean_ms": (perf_counter() - t0) * 10})
        print(f"NLI and selector {index + 1}/24", flush=True)
    (output / "review_cases.jsonl").write_text("\n".join(json.dumps(x) for x in review_cases) + "\n", encoding="utf-8")
    (output / "classification_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    print("Indexing full 5,183-document corpus with real local MiniLM", flush=True)
    papers = [RetrievalPaper(paper_id=str(p["doc_id"]), title=p["title"], abstract=p["abstract"]) for p in corpus]
    t0 = perf_counter()
    retrievers = build_all(papers, encoder=local_encoder(), cache_dir=output / "embedding_cache", corpus="scifact")
    index_seconds = perf_counter() - t0
    claim_map = {str(c["id"]): c for c in claims}
    methods = {m: [] for m in retrievers}
    retrieval_records = []
    for cid in sorted({r["claim_id"] for r in selected}):
        claim = claim_map[cid]
        relevant = {pid for pid, evidence in claim["evidence"].items() if evidence}
        for method, retriever in retrievers.items():
            result = retriever.search(claim["claim"], k=100, claim_id=cid)
            ranked = [r.paper_id for r in result.results]
            rr = next((1 / (i + 1) for i, pid in enumerate(ranked) if pid in relevant), 0)
            dcg = sum(1 / math.log2(i + 2) for i, pid in enumerate(ranked[:10]) if pid in relevant)
            idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
            record = {"claim_id": cid, "method": method, "mrr_at_100": rr,
                "recall_at_10": len(relevant & set(ranked[:10])) / len(relevant),
                "recall_at_100": len(relevant & set(ranked)) / len(relevant),
                "ndcg_at_10": dcg / idcg, "query_ms": result.latency_ms,
                "ranked_ids": ranked}
            methods[method].append(record)
            retrieval_records.append(record)
    summary = {"dataset": "SciFact-dev", "development_only": True, "pairs": len(records),
        "corpus_documents": len(corpus), "claims": len(methods["bm25"]), "index_seconds": index_seconds,
        "oracle_pair_accuracy": sum(r["oracle_prediction"] == r["gold"] for r in records) / len(records),
        "oracle_confusion": {gold: {pred: sum(r["gold"] == gold and r["oracle_prediction"] == pred for r in records)
            for pred in ["SUPPORT", "CONTRADICT", "NEUTRAL"]} for gold in ["SUPPORT", "CONTRADICT"]},
        "selected_window_verdict_agreement": sum(r["selected_window_verdict"] ==
            {"SUPPORT": "SUPPORTED", "CONTRADICT": "CONTRADICTED"}[r["gold"]] for r in records) / len(records),
        "grounding_gate_passed": sum(r["displayable"] for r in records),
        "subset_mean_ms": sum(r["subset_mean_ms"] for r in records) / len(records),
        "retrieval": {m: {metric: sum(r[metric] for r in rs) / len(rs)
            for metric in ["mrr_at_100", "recall_at_10", "recall_at_100", "ndcg_at_10", "query_ms"]}
            for m, rs in methods.items()},
        "elapsed_seconds": perf_counter() - started,
        "limitations": ["Label-stratified small dev sample; not untouched test performance",
            "Oracle document for selector; not a full-chain verdict accuracy metric",
            "Retrieval uses entire corpus and no gold candidate filtering",
            "NLI threshold fixed at owner 0.50; no tuning performed", "Human scientific usefulness not rated"]}
    (output / "retrieval_records.json").write_text(json.dumps(retrieval_records, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
