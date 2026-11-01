"""Prepare offline, then explicitly opt into guarded free-tier Gemini calls."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from paper_search.relevance.cache import Cache
from paper_search.relevance.dataset import (
    MODELS,
    PROMPT,
    Judgment,
    digest,
    prepare,
)
from paper_search.relevance.gemini import (
    EvaluationStopped,
    evaluate,
)
from paper_search.relevance.report import generate_report


def main() -> None:
    """No network activity without both run action and --run-live."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "run", "report")
    )
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "artifacts/arxiv-concepts-20261031/manual_assessment.csv"
        ),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "artifacts/arxiv-concepts-20261031/results.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gemini-relevance-20261031"),
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        help="Choose once during prepare; run reuses the frozen model",
    )
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--daily-limit", type=int, default=20)
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.action == "run" and not args.run_live:
        parser.error("run requires --run-live; no API calls made")
    if (
        args.max_calls < 1
        or args.daily_limit < 1
        or args.interval < 15
    ):
        parser.error("Positive limits and interval >=15 required")
    try:
        if args.action == "prepare":
            manifest = prepare(
                args.source,
                args.results,
                args.output,
                args.model or MODELS[0],
            )
        else:
            manifest = json.loads(
                (args.output / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                manifest["prompt"] != PROMPT
                or (
                    manifest["schema"]
                    != Judgment.model_json_schema()
                )
                or manifest["model"] not in MODELS
            ):
                raise ValueError(
                    "Frozen protocol changed; use a new output"
                )
            if args.model and args.model != manifest["model"]:
                raise ValueError("Cannot change the frozen model")
        lock = args.output / ".run.lock"
        try:
            file = lock.open("x", encoding="utf-8")
        except FileExistsError:
            parser.error(
                "Evaluation locked; do not run two processes"
            )
        try:
            with file:
                file.write(str(os.getpid()))
            cache = Cache(args.output / "cache.sqlite3")
            stopped = None
            try:
                fingerprint = digest(manifest)
                previous = cache.setting("manifest_fingerprint")
                if previous and previous != fingerprint:
                    raise ValueError(
                        "Cached manifest changed; refusing reuse"
                    )
                cache.set_setting(
                    "manifest_fingerprint", fingerprint
                )
                if args.action == "run":

                    async def run() -> int:
                        async with httpx.AsyncClient(
                            timeout=60,
                            follow_redirects=False,
                            trust_env=False,
                        ) as client:
                            return await evaluate(
                                manifest,
                                cache,
                                client,
                                api_key=os.getenv(
                                    "GEMINI_API_KEY", ""
                                ),
                                oauth_token=os.getenv(
                                    "GOOGLE_OAUTH_ACCESS_TOKEN", ""
                                ),
                                max_calls=args.max_calls,
                                daily_limit=args.daily_limit,
                                interval=args.interval,
                            )

                    try:
                        asyncio.run(run())
                    except EvaluationStopped as exc:
                        stopped = str(exc)
                summary = generate_report(
                    manifest, cache, args.output
                )
            finally:
                cache.close()
        finally:
            lock.unlink()
        print(
            f"Prepared {manifest['row_count']} rows / "
            f"{manifest['pair_count']} pairs. "
            f"Judged {summary['judged_pairs']}. Output: {args.output}"
        )
        if stopped:
            parser.exit(2, stopped + "\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Offline setup error: {exc}\n")


if __name__ == "__main__":
    main()
