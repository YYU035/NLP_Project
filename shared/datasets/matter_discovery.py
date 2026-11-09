"""Optional Matter-of-Fact -> Person 1 bridge with explicit date diagnostics.

This performs live search only when the CLI is explicitly executed. Metadata
dates are checked, not independently authenticated publication/version history.
"""

import argparse
import asyncio
from datetime import date
import json
from pathlib import Path


async def discover_case(case: dict, service, *, top_k: int = 30) -> dict:
    if set(case) != {"case_id", "claim_text", "evidence_not_after"}:
        raise ValueError("Expected the leakage-safe prepared inference case only")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be positive")
    cutoff = date.fromisoformat(case["evidence_not_after"])
    claim = case["claim_text"]
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Original claim must be nonempty")
    result = await service.retrieve(claim, top_k=top_k, to_publication_date=cutoff)
    if result.original_claim != claim:
        raise ValueError("Discovery changed the original claim")
    accepted, rejected = [], []
    for entry in result.papers:
        dates = [entry.paper.publication_date,
                 *[record.publication_date for record in entry.source_records]]
        reason = ("unknown_publication_date" if any(value is None for value in dates)
                  else "post_cutoff_record" if any(value > cutoff for value in dates) else None)
        if reason:
            rejected.append({"identity": entry.identity, "reason": reason})
        else:
            accepted.append(entry)
    # Provider records may contain rejected papers. Never pass them as model text.
    candidates = result.model_copy(deep=True)
    candidates.papers = accepted
    for provider in candidates.providers.values():
        provider.records = []
    return {
        "case_id": case["case_id"], "evidence_not_after": cutoff.isoformat(),
        "candidates": candidates.model_dump(mode="json"),
        "audit": {"raw_discovery": result.model_dump(mode="json"),
                  "rejected": rejected, "accepted_count": len(accepted)},
        "qualification": "Metadata-date filter only; exact publication/version history and model knowledge cutoff remain unverified. Empty output is not an infeasible verdict.",
    }


async def run(args):
    from paper_search.retrieval import RetrievalService

    payload = json.loads(args.input.read_text())
    if payload.get("dataset") != "Matter-of-Fact" or not payload.get("version"):
        raise ValueError("Expected a prepared Matter-of-Fact inference file")
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = {"dataset_version": payload["version"], "status": "running", "cases": []}
    try:
        async with RetrievalService() as service:
            for index, case in enumerate(payload["cases"][:args.limit]):
                result = await discover_case(case, service, top_k=args.top_k)
                prefix = f"case-{index:05d}"
                (args.output_dir / f"{prefix}-candidates.json").write_text(json.dumps({
                    "case_id": result["case_id"], "dataset_version": payload["version"],
                    "evidence_not_after": result["evidence_not_after"],
                    "result": result["candidates"], "qualification": result["qualification"],
                }, indent=2) + "\n")
                (args.output_dir / f"{prefix}-audit.json").write_text(json.dumps(result["audit"], indent=2) + "\n")
                manifest["cases"].append({"case_id": result["case_id"], "file": f"{prefix}-candidates.json"})
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=30)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
