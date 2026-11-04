"""Run from repository root: python -m ranking_evaluation.examples.mock_input."""

from paper_search.metadata import ScientificPaper
from paper_search.retrieval_models import (
    PaperOrigin,
    RetrievedPaper,
    RetrievalResult,
)

from ranking_evaluation.schemas import RankingInput


def build_mock_input() -> RankingInput:
    """Build real Person 1 schemas with explicitly fictional metadata/rankings."""
    claim = "Synthetic claim for interface testing only."
    entries = []
    for suffix in ("A", "B", "C"):
        paper = ScientificPaper(
            source="synthetic", source_id=suffix,
            title=f"Synthetic paper {suffix} (not a real publication)",
        )
        entries.append(RetrievedPaper(
            identity=f"synthetic:{suffix}",
            identity_aliases=[f"synthetic:{suffix}"],
            original_claim=claim,
            paper=paper,
            provenance=[PaperOrigin(
                provider="synthetic", source_id=suffix,
                query_id="mock-query", rank=len(entries) + 1,
            )],
            source_records=[paper],
        ))
    return RankingInput(
        claim_id="mock-claim-1",
        candidates=RetrievalResult(original_claim=claim, top_k=3, papers=entries),
        bm25_ranking=["synthetic:A", "synthetic:C", "synthetic:B"],
        sbert_ranking=["synthetic:B", "synthetic:A", "synthetic:C"],
    )


if __name__ == "__main__":
    print(build_mock_input().model_dump_json(indent=2))
