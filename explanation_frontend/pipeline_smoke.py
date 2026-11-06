"""Reproducible real local-model smoke; artifacts stay in ignored outputs/."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from .discovery import sample_discovery, LiveDiscoveryBridge
from .pipeline import LocalPipeline
from .models import make_context
from .templates import build_template
from .validation import validate_explanation
from .export import export_snapshot, import_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--claim", default="Long-term exposure to fine particulate matter is associated with cardiovascular mortality.")
    args = parser.parse_args()
    if not 1 <= args.limit <= 10:
        parser.error("limit must be 1..10")
    started = perf_counter()
    if args.live:
        bridge = LiveDiscoveryBridge()
        try:
            source = bridge.retrieve(args.claim, top_k=args.limit)
        finally:
            bridge.close()
    else:
        source = sample_discovery()
    ids = [p.identity for p in source.result.papers if p.paper.abstract][:args.limit]
    pipeline = LocalPipeline()
    snapshot = pipeline.run_from_discovery(source, ids)
    context = make_context(snapshot, ids, snapshot.provisional)
    explanation = build_template(context)
    gate = validate_explanation(explanation, context)
    if not gate.displayable:
        raise ValueError("Explanation gate failed")
    bundle = export_snapshot(context, explanation)
    assert import_snapshot(bundle).snapshot.fingerprint == snapshot.fingerprint
    before = perf_counter()
    subset = pipeline.aggregate(snapshot, ids[:1])
    subset_ms = (perf_counter() - before) * 1000
    report = {"source": "live" if args.live else "recorded real discovery",
        "claim": snapshot.claim, "candidate_count": len(source.result.papers),
        "verified_papers": len(snapshot.papers), "passages": len(snapshot.verification.evidence_passages),
        "verdict": snapshot.provisional.aggregated_verdict, "status": snapshot.provisional.status,
        "displayable": gate.displayable, "subset_verdict": subset.aggregated_verdict,
        "subset_ms": subset_ms, "elapsed_seconds": perf_counter() - started,
        "model": snapshot.verification.model_settings.model_dump(mode="json"),
        "scientific_quality": "not independently evaluated"}
    directory = Path(__file__).parent / "outputs" / "integration"
    directory.mkdir(parents=True, exist_ok=True)
    name = "live" if args.live else "recorded"
    (directory / f"{name}-run.json").write_text(bundle, encoding="utf-8")
    (directory / f"{name}-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
