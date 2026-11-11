"""Loaders: SciFact corpus/claims, and Person 1's live-search paper records."""
from __future__ import annotations

import json
import re
from pathlib import Path

from research_verifier.retrieval.schemas import Paper


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------- SciFact
def load_scifact_corpus(path: str | Path) -> list[Paper]:
    """corpus.jsonl rows: {doc_id:int, title:str, abstract:[str], structured:bool}"""
    return [
        Paper(paper_id=str(r["doc_id"]), title=r["title"], abstract=list(r["abstract"]),
              source="scifact", text_status="abstract_only")
        for r in _read_jsonl(path)
    ]


def load_scifact_claims(path: str | Path, drop_nei: bool = True) -> list[dict]:
    """Returns [{claim_id, claim, relevant: set[paper_id]}].

    Relevance = docs listed in `evidence` (the standard SciFact/BEIR qrels).
    NEI claims have no evidence docs, so there is nothing to retrieve; they
    are dropped from retrieval evaluation by default (they still matter for
    Person 4's verification eval).
    """
    out = []
    for r in _read_jsonl(path):
        relevant = {str(k) for k in r.get("evidence", {})}
        if drop_nei and not relevant:
            continue
        out.append({"claim_id": str(r["id"]), "claim": r["claim"], "relevant": relevant})
    return out


# ------------------------------------------------------------- Live (Person 1)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list[str]:
    """Light splitter for live abstracts (SciFact abstracts are pre-split).
    Swap for whatever Person 3 uses so sentence indices match."""
    text = " ".join(text.split())
    return [s for s in _SENT_SPLIT.split(text) if s] if text else []


# Field-name fallbacks so this works before Person 1's schema is final.
_ID_KEYS = ("paper_id", "id", "doi", "arxiv_id", "openalex_id", "paperId")
_TITLE_KEYS = ("title",)
_ABS_KEYS = ("abstract", "summary")
_FULL_KEYS = ("full_text", "fulltext", "pdf_text")


def _first(rec: dict, keys) -> object | None:
    for k in keys:
        if rec.get(k):
            return rec[k]
    return None


def paper_from_live_record(rec: dict) -> Paper:
    pid = _first(rec, _ID_KEYS)
    if pid is None:
        raise ValueError(f"Live record has no identifier field: keys={list(rec)}")
    abstract = _first(rec, _ABS_KEYS) or []
    if isinstance(abstract, str):
        abstract = split_sentences(abstract)
    full = _first(rec, _FULL_KEYS)
    if isinstance(full, list):  # e.g. per-page text
        full = "\n".join(p if isinstance(p, str) else p.get("text", "") for p in full)
    status = rec.get("text_status") or rec.get("availability") or (
        "full_text" if full else "abstract_only" if abstract else "unavailable")
    year = rec.get("year") or (str(rec.get("publication_date", ""))[:4] or None)
    return Paper(
        paper_id=str(pid), title=_first(rec, _TITLE_KEYS) or "", abstract=list(abstract),
        source=rec.get("source", "live"), text_status=status,
        year=int(year) if year and str(year).isdigit() else None,
        url=rec.get("url") or rec.get("pdf_url"), full_text=full,
        extra={k: v for k, v in rec.items() if k in ("authors", "venue", "citation_count")},
    )


def load_live_papers(path_or_records) -> list[Paper]:
    """Accepts a JSON/JSONL path or a list of dicts. Dedupes on paper_id
    (Person 1 should already dedupe; this is a guard, not the main logic)."""
    if isinstance(path_or_records, (str, Path)):
        p = Path(path_or_records)
        if p.suffix == ".jsonl":
            records = _read_jsonl(p)
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
            records = data.get("papers", data) if isinstance(data, dict) else data
    else:
        records = path_or_records
    seen, papers = set(), []
    for rec in records:
        paper = paper_from_live_record(rec)
        if paper.paper_id in seen:
            continue
        seen.add(paper.paper_id)
        papers.append(paper)
    return papers
