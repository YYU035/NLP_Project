"""Person 3 application boundary: upstream rankings to downstream paper list."""

from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ranking_evaluation.latency import LatencyRecord, measure_call
from ranking_evaluation.reranker import PairScorer, rerank
from ranking_evaluation.rrf import reciprocal_rank_fusion
from ranking_evaluation.schemas import RankedPaper, RankingInput


class RankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rrf_k: int = Field(default=60, ge=0)
    output_top_k: int | None = Field(default=None, ge=0)
    rerank_enabled: bool = False
    rerank_top_n: int = Field(default=20, ge=0)


class ModelDescription(BaseModel):
    """Caller records the actual scorer configuration; not inferred by ranking."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    device: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    max_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)


class RankingOutput(BaseModel):
    """Serializable Person 4 handoff; no verification verdict is inferred."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    inputs: RankingInput
    configuration: RankingConfig
    model: ModelDescription | None
    papers: list[RankedPaper]
    ranked_union_size: int
    rerank_scored_count: int
    latency: LatencyRecord
    ranking_seconds: float = Field(ge=0, allow_inf_nan=False)
    warnings: list[str]


def rank_claim(
    inputs: RankingInput, *, config: RankingConfig | None = None,
    scorer: PairScorer | None = None, model: ModelDescription | None = None,
) -> RankingOutput:
    """Fuse, optionally rerank, then truncate; preserve original diagnostics.

    Models are constructed by the caller and can be reused across claims.
    ranking_seconds excludes search, retrieval, model loading and JSON output.
    It is not end-to-end system latency. No upstream failure is hidden.
    """
    started = perf_counter()
    inputs = RankingInput.model_validate(inputs.model_dump())
    config = RankingConfig.model_validate((config or RankingConfig()).model_dump())
    if config.rerank_enabled and (scorer is None or model is None):
        raise ValueError("Enabled reranking requires an explicit scorer and model description")
    if not config.rerank_enabled and (scorer is not None or model is not None):
        raise ValueError("Disable reranking without passing a scorer or model description")
    if model is not None:
        model = ModelDescription.model_validate(model.model_dump())
    warnings = []
    if not inputs.bm25_ranking:
        warnings.append("BM25 ranking is empty; inspect upstream retrieval status")
    if not inputs.sbert_ranking:
        warnings.append("SBERT ranking is empty; inspect upstream retrieval status")
    for name, provider in inputs.candidates.providers.items():
        if provider.status not in ("success", "zero_results"):
            warnings.append(f"Upstream provider {name} status: {provider.status}")
    fused, fusion_seconds = measure_call(lambda: reciprocal_rank_fusion(
        inputs, rrf_k=config.rrf_k,
    ))
    union_size = len(fused)
    rerank_seconds, scored_count = None, 0
    if config.rerank_enabled:
        fused, rerank_seconds = measure_call(lambda: rerank(
            fused, claim=inputs.candidates.original_claim, scorer=scorer,
            top_n=config.rerank_top_n,
        ))
        prefix = fused[:config.rerank_top_n]
        scored_count = sum(row.cross_encoder_score is not None for row in prefix)
        skipped = len(prefix) - scored_count
        if skipped:
            warnings.append(f"{skipped} rerank-prefix papers lacked title and abstract; kept their slots")
    result = fused[:config.output_top_k]
    return RankingOutput(
        inputs=inputs, configuration=config, model=model, papers=result,
        ranked_union_size=union_size, rerank_scored_count=scored_count,
        latency=LatencyRecord(
            bm25_seconds=inputs.bm25_seconds, sbert_seconds=inputs.sbert_seconds,
            rrf_seconds=fusion_seconds, rerank_seconds=rerank_seconds,
        ),
        ranking_seconds=perf_counter() - started, warnings=warnings,
    )
