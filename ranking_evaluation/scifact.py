"""SciFact train/dev adapter; evidence-bearing documents define relevance."""

import argparse
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt
from paper_search.metadata import ScientificPaper
from paper_search.retrieval_models import RetrievedPaper, RetrievalResult

from ranking_evaluation.benchmark import BenchmarkBundle
from ranking_evaluation.schemas import RankingInput


class CorpusDocument(BaseModel):
    doc_id: StrictInt
    title: str
    abstract: list[str]
    structured: bool


class EvidenceSet(BaseModel):
    label: Literal["SUPPORT", "CONTRADICT"]
    sentences: list[StrictInt] = Field(min_length=1)


class Claim(BaseModel):
    id: StrictInt
    claim: str = Field(min_length=1)
    evidence: dict[str, list[EvidenceSet]]  # Required: absent labels are not NEI.
    cited_doc_ids: list[StrictInt]


class RankingExportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    bm25_ranking: list[str]
    sbert_ranking: list[str]
    bm25_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    sbert_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class RankingExport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"] = "1"
    status: Literal["pending", "complete"]
    corpus_sha256: str
    claims_sha256: str
    retrieval_config: dict[str, str] = Field(min_length=1)
    records: list[RankingExportRecord]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_scifact(corpus_path: Path, claims_path: Path, *, split: str):
    """Parse and validate full split, including sentence references and IDs."""
    if split not in ("train", "dev"):
        raise ValueError("Only labeled train/dev splits are supported")
    docs, claims = {}, {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = CorpusDocument.model_validate_json(line)
        identity = str(doc.doc_id)
        if identity in docs:
            raise ValueError("Duplicate corpus document ID")
        docs[identity] = doc
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        claim = Claim.model_validate_json(line)
        identity = str(claim.id)
        if identity in claims:
            raise ValueError("Duplicate claim ID")
        if {str(doc_id) for doc_id in claim.cited_doc_ids} - docs.keys():
            raise ValueError("Cited document is absent from corpus")
        for doc_id, sets in claim.evidence.items():
            if doc_id not in docs or not sets:
                raise ValueError("Evidence must reference a corpus document and nonempty rationale sets")
            for rationale in sets:
                if any(index < 0 or index >= len(docs[doc_id].abstract)
                       for index in rationale.sentences):
                    raise ValueError("Evidence sentence index is out of range")
        claims[identity] = claim
    if not docs or not claims:
        raise ValueError("Corpus and claim split must not be empty")
    return docs, claims


def convert_scifact(
    corpus_path: Path, claims_path: Path, rankings: RankingExport, *, split: str,
) -> BenchmarkBundle:
    """Join Person 2 rankings with SciFact text; never rank using gold labels."""
    rankings = RankingExport.model_validate(rankings.model_dump())
    if rankings.status != "complete":
        raise ValueError("Pending ranking template cannot be evaluated")
    corpus_hash, claims_hash = sha256(corpus_path), sha256(claims_path)
    if rankings.corpus_sha256 != corpus_hash or rankings.claims_sha256 != claims_hash:
        raise ValueError("Ranking export data hashes do not match SciFact files")
    docs, claims = load_scifact(corpus_path, claims_path, split=split)
    records = {row.claim_id: row for row in rankings.records}
    if len(records) != len(rankings.records) or records.keys() != claims.keys():
        raise ValueError("Ranking export must cover each split claim exactly once")
    inputs = []
    for claim_id, claim in claims.items():
        record = records[claim_id]
        candidate_ids = list(dict.fromkeys(record.bm25_ranking + record.sbert_ranking))
        if set(candidate_ids) - docs.keys():
            raise ValueError("Ranked document is absent from corpus")
        entries = []
        for doc_id in candidate_ids:
            doc = docs[doc_id]
            paper = ScientificPaper(
                source="scifact", source_id=doc_id, title=doc.title or None,
                abstract=" ".join(doc.abstract) or None,
            )
            entries.append(RetrievedPaper(
                identity=doc_id, identity_aliases=[doc_id], original_claim=claim.claim,
                paper=paper, provenance=[], source_records=[paper],
            ))
        inputs.append(RankingInput(
            claim_id=claim_id,
            candidates=RetrievalResult(
                original_claim=claim.claim, papers=entries,
                top_k=max(len(record.bm25_ranking), len(record.sbert_ranking)),
            ),
            bm25_ranking=record.bm25_ranking, sbert_ranking=record.sbert_ranking,
            bm25_seconds=record.bm25_seconds, sbert_seconds=record.sbert_seconds,
        ))
    return BenchmarkBundle(
        dataset_name="SciFact", dataset_version=f"corpus:{corpus_hash};claims:{claims_hash}",
        split=split, synthetic=False,
        relevance_definition="Binary relevant = document with nonempty annotated SUPPORT or CONTRADICT evidence; cited_doc_ids alone are not positives; empty evidence excluded from relevance macro means",
        retrieval_config={**rankings.retrieval_config,
                          "candidate_origin": "fixed corpus; union of supplied branch rankings"},
        corpus_doc_ids=list(docs), claims=inputs,
        qrels={identity: list(claim.evidence) for identity, claim in claims.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    if args.rankings:
        result = convert_scifact(
            args.corpus, args.claims,
            RankingExport.model_validate_json(args.rankings.read_bytes()), split=args.split,
        )
    else:
        docs, claims = load_scifact(args.corpus, args.claims, split=args.split)
        result = RankingExport(
            status="pending", corpus_sha256=sha256(args.corpus),
            claims_sha256=sha256(args.claims),
            retrieval_config={"status": "Person 2 must fill actual methods/models/settings"},
            records=[RankingExportRecord(claim_id=identity, bm25_ranking=[], sbert_ranking=[])
                     for identity in claims],
        )
        print(f"Validated {len(docs)} documents and {len(claims)} claims; writing pending template")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
