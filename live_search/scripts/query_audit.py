"""Compare captured pre-audit outputs with the current generator, locally."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import spacy
from query_audit_cases import CASES

from paper_search.query_processing import DeterministicQueryGenerator
from paper_search.query_rendering import render_query

ROOT = Path(__file__).resolve().parents[1]
_RISKS = {
    "biomedical": "The older-adult population and exact TNF-α/COVID-19 "
    "terminology remain restrictive; backend symbol tokenization may differ.",
    "immunology": "No alternative emitted. Semantic Scholar still cannot "
    "enforce exact hyphenated identifier matching.",
    "battery_engineering": "The retained 90%, 500-cycle and -5 °C conditions "
    "may exclude papers reporting other units or experimental settings.",
    "power_electronics": "The silicon comparator and 100 kHz condition "
    "remain restrictive; comparison language is not interpreted logically.",
    "climate": "Coastal waters remains a population/environment constraint; "
    "relevant open-ocean studies may not match.",
    "astronomy": "No alternative emitted. Terminology variation is not "
    "addressed through unsupported synonyms.",
    "agriculture": "Both drought conditions and the without-decreasing "
    "qualification are retained; a paper may express them differently.",
    "nutrition_uncertainty": "No alternative emitted: the preliminary, "
    "suggest, and may qualifiers must not be erased.",
    "computer_science": "The random-forest comparator remains restrictive; "
    "removing reporting terms can change the provider's capped result order.",
    "catalysis": "The 25 °C and acidic-solution conditions remain "
    "restrictive; related studies may describe different temperatures "
    "or environments.",
    "mathematics": "No alternative emitted. Universal quantifiers and the "
    "compact-interval condition remain intact.",
    "general_science": "No alternative emitted: the original is already "
    "short, and generic noun lists add no useful focus.",
    "chemistry_numbers": "No alternative emitted. Scientific exponents, "
    "negative/polarity context, and pH values remain provider-sensitive.",
    "biomedical_population": "The under-five age constraint remains "
    "restrictive. General English NER may mislabel the age span as DATE.",
}


def main() -> None:
    """Write exact before/after strings and qualitative, unmeasured risks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/query-audit.json")
    parser.add_argument("--markdown", type=Path,
                        default=ROOT / "docs/query_audit.md")
    args = parser.parse_args()
    snapshot = json.loads(
        (ROOT / "docs/query_audit_before.json").read_text(encoding="utf-8")
    )
    if [(c["domain"], c["claim"]) for c in snapshot["examples"]] != CASES:
        raise ValueError(
            "Audit claims must match the captured baseline exactly"
        )
    generator = DeterministicQueryGenerator()
    cases = []
    for old in snapshot["examples"]:
        details = generator.generate_with_details(old["claim"])
        cases.append({
            **old, "after": details.model_dump(),
            "qualitative_recall_risk": _RISKS[old["domain"]],
            "provider_rendering": [
                {name: render_query(query, name).model_dump() for name in (
                    "openalex", "arxiv", "semantic_scholar"
                )} for query in details.queries
            ],
        })
    built = sum(len(case["after"]["queries"]) == 2 for case in cases)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "baseline_captured_at_utc": snapshot["captured_at_utc"],
        "spacy_version": spacy.__version__,
        "retrieval_evaluated": False,
        "summary": {"claims": len(cases), "focused": built,
                    "original_only": len(cases) - built},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
    lines = [
        "# Scientific query audit: actual before/after outputs", "",
        "Captured baseline outputs are in `query_audit_before.json`; the old",
        "generator is not reimplemented by this report. These strings were",
        "generated locally using spaCy " + spacy.__version__ + ".", "",
        f"**{len(cases)} claims: {built} focused alternatives, "
        f"{len(cases) - built} original-only decisions.**", "",
        "The first query after this audit is the original input unchanged.",
        "No retrieval evaluation was performed. Risks below are qualitative,",
        "not measured recall losses. Retained conditions are existing claim",
        "constraints, not constraints introduced by the focused query.", "",
        "Removing reporting words can broaden matches and change provider",
        "ordering; under a fixed top-k budget, this can still lose relevant",
        "papers from the returned list. No recall guarantee follows from",
        "shortening the query. Exact symbol matching is backend-dependent.",
        "",
    ]
    for number, case in enumerate(cases, 1):
        old = case["before_queries"]
        new = case["after"]["queries"]
        lines.extend([
            f"## {number}. {case['domain']}", "",
            "Original / primary query:", "", "```text", case["claim"],
            "```", "", "Before: additional focused query:", "", "```text",
            old[1] if len(old) > 1 else "[none]", "```", "",
            "After: additional focused query:", "", "```text",
            new[1] if len(new) > 1 else "[none — original query only]",
            "```", "",
            "Decision: `" + case["after"]["construction"]["reason"] + "`.",
            "", "Selected concepts: " + "; ".join(
                c["text"] for c in
                case["after"]["construction"]["selected_concepts"]
            ) + ".", "", "Recall caveat: " + case["qualitative_recall_risk"],
            "",
        ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["summary"]))
    print(f"JSON: {args.output}")
    print(f"Before/after: {args.markdown}")


if __name__ == "__main__":
    main()
