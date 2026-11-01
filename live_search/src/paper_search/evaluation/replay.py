"""Replay confirmed sparse-empty envelopes without repeating live requests."""

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx

from paper_search.clients.semantic_scholar import (
    SemanticScholarClient,
)
from paper_search.evaluation.models import ExperimentOptions
from paper_search.evaluation.runner import metadata_profile


def reparse_sparse_empty_responses(
    manifest: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Keep raw observations intact and annotate any corrected classifications.

    Only archived HTTP-200 responses with the exact observed empty envelope
    qualify. Validate the hash and execute the production client with an
    httpx.MockTransport. Request counts and live timings are never changed.
    """
    result = deepcopy(manifest)
    options = ExperimentOptions.model_validate(result["options"])
    for call in result["calls"]:
        if (
            call["provider"] != "semantic_scholar"
            or call["status"] in {"success", "zero_results"}
            or not call["attempts"]
        ):
            continue
        attempt = call["attempts"][-1]
        if attempt["status"] != 200 or not attempt.get(
            "response_file"
        ):
            continue
        body = (output / attempt["response_file"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != attempt["sha256"]:
            raise ValueError("Archived response hash mismatch")
        try:
            payload = json.loads(body)
        except ValueError:
            continue
        if payload != {"total": 0, "offset": 0}:
            continue

        async def replay(query: str, saved_body: bytes) -> None:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        200, content=saved_body
                    )
                )
            ) as http:
                papers = await SemanticScholarClient(http).search(
                    query,
                    **options.model_dump(
                        include={
                            "max_results",
                            "from_publication_date",
                            "to_publication_date",
                        }
                    ),
                )
                if papers:
                    raise ValueError(
                        "Empty-envelope replay returned papers"
                    )

        asyncio.run(replay(call["query"], body))
        call["parser_correction"] = {
            "observed_status": call["status"],
            "observed_error": call["error"],
            "observed_result_count": call["result_count"],
            "reason": "Verified HTTP 200 zero-total envelope without data",
            "verification": "offline production-client replay; no live I/O",
            "response_file": attempt["response_file"],
        }
        call.update(
            {
                "status": "zero_results",
                "result_count": 0,
                "raw_result_count": 0,
                "error": None,
                "cause_type": None,
                "papers": [],
                "metadata": metadata_profile([], options),
            }
        )
    return result
