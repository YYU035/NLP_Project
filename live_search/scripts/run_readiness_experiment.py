"""Future opt-in depth or arXiv-fallback run; not run during preparation."""

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

import httpx

from paper_search.arxiv_fallback import relaxation_plan
from paper_search.clients.arxiv import ArxivClient
from paper_search.clients.openaire import OpenAIREClient
from paper_search.clients.openalex import OpenAlexClient
from paper_search.config import Settings
from paper_search.evaluation.depth import prefix_summary
from paper_search.retrieval import RetrievalService, RetrievalUnavailable
from paper_search.retrieval_models import ProviderPlan


def factory(http: httpx.AsyncClient, settings: Settings) -> dict:
    """One attempt/query; preserve provider pacing."""
    clients = {}
    for name, cls, secret in (
        ("openalex", OpenAlexClient, settings.openalex_api_key),
        ("openaire", OpenAIREClient, settings.openaire_token),
        ("arxiv", ArxivClient, None),
    ):
        if getattr(settings, f"{name}_enabled"):
            clients[name] = cls(
                http,
                max_retries=0,
                api_key=secret.get_secret_value() if secret else None,
                timeout_seconds=settings.timeout_seconds,
            )
    return clients


async def run(plan_path: Path, output: Path, mode: str) -> None:
    """Save each completed claim; stop each provider on any failure.

    No automatic resume: use a fresh run directory to prevent accidental extra
    requests. Response bodies and actual HTTP parameters are archived. All new
    candidates remain unlabelled until assessed separately.
    """
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["status"] != "prepared_not_run" or plan["top_k"] != (
        30 if mode == "depth" else 10
    ):
        raise ValueError("Wrong experiment plan")
    depth = mode == "depth"
    if depth:
        rows = plan["plans"]
    else:
        # Reuse frozen concepts from the companion depth plan.
        companion = json.loads(
            (plan_path.parent / "depth_plan.json").read_text(encoding="utf-8")
        )
        rows = [p for p in companion["plans"] if p["provider"] == "arxiv"]
        by_claim = {p["claim_id"]: p for p in plan["plans"]}
        for row in rows:
            if row["query"] != by_claim[row["claim_id"]]["primary"]:
                raise ValueError(
                    "Fallback primary differs from frozen control"
                )
            expected = by_claim[row["claim_id"]]
            actual = relaxation_plan(
                row["required_phrases"], [], max_attempts=plan["max_fallbacks"]
            )
            if [q.query for q in actual] != [
                q["query"] for q in expected["attempts"]
            ]:
                raise ValueError("Fallback renderer differs from frozen plan")
    lookup = {(p["provider"], p["claim"]): p for p in rows}
    providers = tuple(dict.fromkeys(p["provider"] for p in rows))
    settings = Settings.from_environment().model_copy(
        update={
            f"{name}_enabled": name in providers
            for name in ("openalex", "openaire", "arxiv", "semantic_scholar")
        }
    )

    def planner(provider: str, claim: str) -> ProviderPlan:
        row = lookup[(provider, claim)]
        return ProviderPlan(
            **{
                k: row[k]
                for k in ("provider", "strategy", "query", "required_phrases")
            }
        )

    output.mkdir(parents=True)
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    stopped = set()
    summaries = []
    async with RetrievalService(
        settings,
        providers=providers,
        planner=planner,
        factory=factory,
        archive=output,
        arxiv_fallback=not depth,
    ) as service:
        claims = list(dict.fromkeys(p["claim"] for p in rows))
        for claim in claims:
            service.providers = tuple(p for p in providers if p not in stopped)
            if not service.providers:
                break
            try:
                result = await service.retrieve(
                    claim,
                    top_k=plan["top_k"],
                    from_publication_date=date.fromisoformat(
                        plan["from_publication_date"]
                    ),
                    to_publication_date=date.fromisoformat(
                        plan["to_publication_date"]
                    ),
                )
            except RetrievalUnavailable as exc:
                result = exc.result
            claim_id = next(p["claim_id"] for p in rows if p["claim"] == claim)
            (output / f"{claim_id}.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            for provider, outcome in result.providers.items():
                if outcome.status in {
                    "provider_failure",
                    "partial_failure",
                    "query_generation_failure",
                }:
                    stopped.add(provider)
                if depth and outcome.status in {"success", "zero_results"}:
                    for k in plan["prefixes"]:
                        # Retrieval does not create labels; reuse source IDs
                        # and identity_aliases during offline assessment.
                        summaries.append(
                            {
                                "claim_id": claim_id,
                                "provider": provider,
                                **prefix_summary(
                                    [r.paper for r in outcome.records],
                                    k,
                                    lambda _: None,
                                ),
                                "shared_top30_seconds": outcome.seconds,
                                "shared_http_attempts": outcome.http_attempts,
                                "incremental_prefix_http_attempts": 0,
                            }
                        )
    (output / "prefixes_unassessed.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    (output / "stopped_providers.json").write_text(
        json.dumps(sorted(stopped)), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("depth", "fallback"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-live", action="store_true")
    args = parser.parse_args()
    if not args.run_live:
        parser.error("No calls made. Pending experiment requires --run-live")
    asyncio.run(run(args.plan, args.output, args.mode))
