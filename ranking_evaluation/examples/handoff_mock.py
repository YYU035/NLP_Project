"""Write a complete, synthetic Person 3 -> Person 4 JSON handoff."""

import argparse
from pathlib import Path

from ranking_evaluation.examples.mock_input import build_mock_input
from ranking_evaluation.pipeline import RankingConfig, rank_claim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = rank_claim(build_mock_input(), config=RankingConfig(output_top_k=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(result.model_dump_json(indent=2) + "\n")
    print(args.output)
