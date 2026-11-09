"""CSFCube paper-pool ranking adapter; no claim-verification labels invented."""

import argparse
import hashlib
import json
from pathlib import Path

from shared.datasets.setup import verify_download

FACETS = ("background", "method", "result")
SPLITS = ("fold1_dev", "fold1_test", "fold2_dev", "fold2_test", "all")


def load(directory: Path, *, split: str = "fold1_dev", facet: str = "all") -> dict:
    manifest = verify_download(directory, "csfcube")
    if split not in SPLITS or facet not in (*FACETS, "all"):
        raise ValueError("Unknown CSFCube split or facet")
    documents = {}
    for line in (directory / "abstracts-csfcube-preds.jsonl").read_text().splitlines():
        row = json.loads(line)
        identity = row["paper_id"]
        if not isinstance(identity, str) or identity in documents:
            raise ValueError("Duplicate or invalid paper ID")
        if (not isinstance(row["title"], str) or not isinstance(row["abstract"], list)
                or not all(isinstance(s, str) for s in row["abstract"])):
            raise ValueError("Invalid title or abstract")
        documents[identity] = row
    splits = json.loads((directory / "evaluation_splits.json").read_text())
    selection = None if split == "all" else set(splits[facet][split])
    queries, grades = {}, {}
    for current in FACETS if facet == "all" else (facet,):
        pools = json.loads((directory / f"test-pid2anns-csfcube-{current}.json").read_text())
        for paper_id, pool in pools.items():
            query_id = f"{paper_id}_{current}"
            if selection is not None and query_id not in selection:
                continue
            candidates, labels = pool["cands"], pool["relevance_adju"]
            if len(candidates) != len(labels) or len(candidates) != len(set(candidates)):
                raise ValueError("Invalid candidate/grade alignment")
            if paper_id not in documents or set(candidates) - documents.keys():
                raise ValueError("Unknown document ID")
            if any(type(g) is not int or g not in range(4) for g in labels):
                raise ValueError("Expected adjudicated relevance grades 0..3")
            # Some released pools contain the query paper itself. Keep the raw
            # files intact but exclude that trivial match in this declared protocol.
            filtered = [(pid, grade) for pid, grade in zip(candidates, labels) if pid != paper_id]
            candidates, labels = [p for p, _ in filtered], [g for _, g in filtered]
            paper = documents[paper_id]
            text = f"Research facet: {current}\n{paper['title']}\n" + " ".join(paper["abstract"])
            queries[query_id] = {"query_id": query_id, "paper_id": paper_id,
                                 "facet": current, "text": text, "candidate_ids": candidates}
            grades[query_id] = dict(zip(candidates, labels))
    if not queries or (selection is not None and set(queries) != selection):
        raise ValueError("Missing selected query/facet tasks")
    protocol = {"split": split, "facet": facet,
                "query_policy": "explicit facet name plus unchanged full title and abstract",
                "candidate_scope": "released per-query judged pool",
                "judgment": "relevance_adju", "self_match_policy": "exclude query paper from pool and judgments"}
    version = hashlib.sha256(json.dumps({"manifest": manifest, "protocol": protocol}, sort_keys=True).encode()).hexdigest()
    return {"dataset": "CSFCube", "version": version, "manifest": manifest,
            "protocol": protocol, "documents": documents, "queries": queries, "grades": grades}


def prepare(dataset: dict, output: Path) -> None:
    """Keep judgments separate from model-facing retrieval inputs."""
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"dataset": dataset["dataset"], "version": dataset["version"],
              "protocol": dataset["protocol"], "documents": dataset["documents"],
              "queries": dataset["queries"]}
    (output / "retrieval-input.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (output / "judgments.json").write_text(json.dumps({
        "version": dataset["version"], "grades": dataset["grades"],
    }, indent=2) + "\n")
    (output / "rankings-template.json").write_text(json.dumps({
        "status": "pending", "dataset_version": dataset["version"],
        "method": "replace with actual method/settings", "rankings": {
            identity: [] for identity in dataset["queries"]},
    }, indent=2) + "\n")


def bm25_rankings(dataset: dict) -> dict:
    """Use Person 2's actual BM25, with no gold labels provided to scoring."""
    from retrieval.bm25 import BM25Retriever
    from retrieval.schema import Paper

    rankings = {}
    for identity, query in dataset["queries"].items():
        ids = query["candidate_ids"]
        papers = [Paper(paper_id=pid, title=dataset["documents"][pid]["title"],
                        abstract=dataset["documents"][pid]["abstract"], source="csfcube") for pid in ids]
        scorer = BM25Retriever(stemmer="simple")
        scorer.index(papers)
        scores = scorer.score_all(query["text"])
        if len(scores) != len(ids):
            raise ValueError("BM25 returned a different corpus size")
        order = sorted(range(len(ids)), key=lambda index: (-float(scores[index]), index))
        rankings[identity] = [ids[index] for index in order]
    return {"status": "complete", "dataset_version": dataset["version"],
            "method": "Person 2 BM25; k1=0.9,b=0.4; simple stemmer; exact released pools; pool-order ties",
            "rankings": rankings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/csfcube"))
    parser.add_argument("--split", choices=SPLITS, default="fold1_dev")
    parser.add_argument("--facet", choices=[*FACETS, "all"], default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bm25", action="store_true")
    args = parser.parse_args()
    dataset = load(args.data, split=args.split, facet=args.facet)
    prepare(dataset, args.output_dir)
    if args.bm25:
        result = bm25_rankings(dataset)
        (args.output_dir / "bm25-rankings.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"documents": len(dataset["documents"]), "queries": len(dataset["queries"]),
                      "version": dataset["version"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
