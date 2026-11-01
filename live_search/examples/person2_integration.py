"""Minimal live handoff: python examples/person2_integration.py "your claim".

This performs retrieval when executed. Importing it makes no network calls.
"""

import argparse
import asyncio

from paper_search.retrieval import RetrievalUnavailable, retrieve


async def main(claim: str) -> None:
    """Pass candidate metadata to Person 2 while retaining the full result."""
    try:
        result = await retrieve(claim, top_k=30)
    except RetrievalUnavailable as exc:
        # Complete outage is distinct from a valid empty candidate pool.
        result = exc.result
        print(result.model_dump_json(indent=2))
        raise SystemExit(1) from exc

    # Person 2 can rank entry.paper.title / entry.paper.abstract against
    # result.original_claim. Retain identity, aliases and all provenance.
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim", help="Original scientific claim")
    asyncio.run(main(parser.parse_args().claim))
