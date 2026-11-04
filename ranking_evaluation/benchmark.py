"""Fixed benchmark interchange format; no live search or implicit labels."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ranking_evaluation.schemas import RankingInput


class BenchmarkBundle(BaseModel):
    """Exported rankings and gold IDs from one fixed corpus and claim split.

    Corpus IDs include all documents, not just a candidate union. A SciFact
    converter must explicitly choose and record its relevance mapping first.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split: str = Field(min_length=1)
    relevance_definition: str = Field(min_length=1)
    synthetic: bool
    retrieval_config: dict[str, str] = Field(min_length=1)
    corpus_doc_ids: list[str]
    claims: list[RankingInput] = Field(min_length=1)
    qrels: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        corpus = set(self.corpus_doc_ids)
        if len(corpus) != len(self.corpus_doc_ids) or "" in corpus:
            raise ValueError("Corpus IDs must be unique and nonempty")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Claim IDs must be unique")
        if set(self.qrels) != set(claim_ids):
            raise ValueError("qrels must cover exactly every benchmark claim")
        for gold in self.qrels.values():
            if len(gold) != len(set(gold)) or set(gold) - corpus:
                raise ValueError("Gold IDs must be unique and belong to the corpus")
        for claim in self.claims:
            if {entry.identity for entry in claim.candidates.papers} - corpus:
                raise ValueError("Candidate IDs must belong to the corpus")
        return self
