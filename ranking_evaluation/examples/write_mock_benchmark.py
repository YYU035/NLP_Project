"""Write a fictional benchmark bundle for exercising the runner, not research."""

import argparse
from pathlib import Path

from ranking_evaluation.benchmark import BenchmarkBundle
from ranking_evaluation.examples.mock_input import build_mock_input


def build_mock_benchmark() -> BenchmarkBundle:
    claim = build_mock_input()
    return BenchmarkBundle(
        dataset_name="synthetic-interface-check", dataset_version="1",
        split="synthetic", synthetic=True,
        relevance_definition="Manually invented B and unretrieved D relevant; no scientific meaning",
        retrieval_config={"bm25": "manual mock ranking", "sbert": "manual mock ranking"},
        corpus_doc_ids=["synthetic:A", "synthetic:B", "synthetic:C", "synthetic:D"],
        claims=[claim], qrels={claim.claim_id: ["synthetic:B", "synthetic:D"]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(build_mock_benchmark().model_dump_json(indent=2) + "\n")
