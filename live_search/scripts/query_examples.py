"""Write real local-model queries and diagnostics; perform no retrieval."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import spacy

from paper_search.query_processing import DeterministicQueryGenerator
from paper_search.query_rendering import render_query

EXAMPLES = [
    ("biomedical", "The expression of TNF-α does not increase the risk of "
     "COVID-19 in older adults."),
    ("engineering", "Lithium-ion batteries retain 90% capacity after 500 "
     "cycles at -5 °C."),
    ("general_science", "Regular physical activity may improve working "
     "memory in older adults."),
    ("engineering_comparison", "The efficiency of SiC MOSFETs is not lower "
     "than that of silicon devices at 100 kHz."),
    ("biomedical_population", "Vitamin A supplementation does not reduce "
     "mortality in children under five years of age."),
    ("no_noun_concepts", "Run quickly."),
]


def main() -> None:
    """Record both ablations, backend rendering, and failure observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/linguistic-query-report.json"),
    )
    args = parser.parse_args()
    generator = DeterministicQueryGenerator(mode="original_plus_concepts")
    control = DeterministicQueryGenerator(mode="original_only")
    examples = []
    for domain, claim in EXAMPLES:
        details = generator.generate_with_details(claim)
        examples.append({
            "domain": domain, "claim": claim,
            "original_only": control.generate(claim),
            "original_plus_concepts": details.model_dump(),
            "provider_rendering": [{
                "canonical_query": query,
                "providers": {
                    provider: render_query(query, provider).model_dump()
                    for provider in ("openalex", "arxiv", "semantic_scholar")
                },
            } for query in details.queries],
        })
        print(json.dumps({
            "domain": domain, "queries": details.queries,
            "extraction_status": details.extraction.status,
            "warnings": details.extraction.warnings,
        }, ensure_ascii=True))
    missing = DeterministicQueryGenerator(
        model="deliberately_unavailable_model"
    ).generate_with_details(EXAMPLES[0][1])
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "spacy_version": spacy.__version__,
        "retrieval_evaluated": False,
        "examples": examples,
        "missing_model_demonstration": missing.model_dump(),
        "limitations": [
            "General English POS/NER can misparse scientific entities.",
            "Dependency labels do not validate outcome/population roles.",
            "Reformulation can change top-k recall; not evaluated here.",
            "Semantic Scholar relevance search cannot enforce phrases; "
            "internal letter hyphens render as spaces with explicit warnings.",
            "Provider indexes control scientific symbol matching.",
            "No negation-scope or scientific claim truth validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
