"""Opt-in live coverage experiment or offline regeneration of saved results."""

import argparse
import asyncio
import json
from datetime import UTC, date, datetime
from pathlib import Path

from paper_search.config import Settings
from paper_search.evaluation.dataset import CLAIMS
from paper_search.evaluation.models import ExperimentOptions
from paper_search.evaluation.replay import (
    reparse_sparse_empty_responses,
)
from paper_search.evaluation.reporting import write_results
from paper_search.evaluation.runner import run_experiment


def main() -> None:
    """Require explicit live opt-in; save artifacts even for degraded runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--from-date", type=date.fromisoformat)
    parser.add_argument("--to-date", type=date.fromisoformat)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--no-matched-budget", action="store_true")
    parser.add_argument(
        "--summarize-existing",
        type=Path,
        help="Regenerate reports offline from an experiment directory",
    )
    args = parser.parse_args()
    if args.summarize_existing:
        output = args.summarize_existing
        manifest = json.loads(
            (output / "run.json").read_text(encoding="utf-8")
        )
        manifest["calls"] = [
            json.loads(line)
            for line in (output / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        # The finalized copy carries the actual end time when available.
        if (output / "results.json").exists():
            previous = json.loads(
                (output / "results.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest["finished_at_utc"] = previous.get(
                "finished_at_utc"
            )
        manifest = reparse_sparse_empty_responses(manifest, output)
        write_results(manifest, output)
        print(f"Regenerated: {output / 'report.md'}")
        return
    if not args.run_live:
        parser.error(
            "Live requests require --run-live; no network calls were made"
        )
    settings = Settings.from_environment()
    if args.timeout is not None:
        settings = Settings.model_validate(
            {
                **settings.model_dump(),
                "timeout_seconds": args.timeout,
            }
        )
    options = ExperimentOptions(
        max_results=args.limit,
        from_publication_date=args.from_date,
        to_publication_date=args.to_date,
        matched_budget=not args.no_matched_budget,
    )
    output = args.output_dir or Path("artifacts") / (
        "query-evaluation-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    manifest = asyncio.run(
        run_experiment(CLAIMS, settings, options, output)
    )
    results = write_results(manifest, output)
    print(json.dumps(results["provider_summary"], indent=2))
    print(f"Report: {(output / 'report.md').resolve()}")
    if any(
        s["failures"] for s in results["provider_summary"].values()
    ):
        raise SystemExit(
            1
        )  # A degraded experiment is never called fully passing.


if __name__ == "__main__":
    main()
