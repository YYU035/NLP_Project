"""Optional cross-encoder scoring with no implicit download or heavy import."""

from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import Protocol

from ranking_evaluation.schemas import RankedPaper


class PairScorer(Protocol):
    """Return one scalar relevance score per (original claim, paper text)."""

    def score_pairs(self, pairs: list[tuple[str, str]]) -> Sequence[float]: ...


class LocalCrossEncoder:
    """Explicitly load a previously downloaded local model; CPU by default."""

    def __init__(
        self, model_path: str | Path, *, device: str = "cpu",
        max_length: int = 512, batch_size: int = 16, cpu_float64: bool = False,
    ):
        if not Path(model_path).is_dir():
            raise ValueError("model_path must be an existing local model directory")
        for name, value in (("max_length", max_length), ("batch_size", batch_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Install ranking_evaluation/requirements-rerank.txt first") from exc
        self.model = CrossEncoder(
            str(Path(model_path).resolve()), device=device,
            max_length=max_length, local_files_only=True, trust_remote_code=False,
        )
        if cpu_float64:
            if device != "cpu":
                raise ValueError("cpu_float64 requires device='cpu'")
            self.model.model.double()
        self._synchronize = None
        if device.startswith("cuda"):
            import torch
            self._synchronize = lambda: torch.cuda.synchronize(device)
        self.batch_size = batch_size

    def score_pairs(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        synchronize = getattr(self, "_synchronize", None)
        if synchronize:
            synchronize()
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True,
        )
        if synchronize:
            synchronize()
        # Library versions may retain a single-label dimension (N, 1).
        if getattr(scores, "ndim", 0) == 2 and scores.shape[1] == 1:
            return scores[:, 0]
        return scores


def rerank(
    papers: Sequence[RankedPaper], *, claim: str, scorer: PairScorer | None = None,
    enabled: bool = True, top_n: int = 20,
) -> list[RankedPaper]:
    """Reorder only the first top_n fused papers, retaining all remaining rows.

    Missing-text entries keep their slots within the prefix, with no model
    score. Other prefix entries exchange only the scoreable slots, sorted by
    model score descending, breaking ties by their incoming RRF position.
    Exceptions and invalid model output propagate, never silently falling back.
    Scores from an earlier rerank are cleared in all returned entries.
    """
    if type(top_n) is not int or top_n < 0:
        raise ValueError("top_n must be a non-negative integer")
    rows = [RankedPaper.model_validate(row.model_dump()) for row in papers]
    ids = [row.entry.identity for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Paper identities must be unique")
    if [row.rank for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Input ranks must be consecutive and ordered from 1")
    if any(row.entry.original_claim != claim for row in rows):
        raise ValueError("claim must match every original claim")
    for row in rows:
        row.cross_encoder_score = None
    if not enabled or top_n == 0 or not rows:
        return rows
    if scorer is None:
        raise ValueError("An explicit scorer is required when reranking is enabled")
    pairs, slots = [], []
    for index, row in enumerate(rows[:top_n]):
        text = "\n\n".join(part.strip() for part in (
            row.entry.paper.title, row.entry.paper.abstract,
        ) if part and part.strip())
        if text:
            slots.append(index)
            pairs.append((claim, text))
    if not pairs:
        return rows
    raw_scores = list(scorer.score_pairs(pairs))
    if len(raw_scores) != len(pairs):
        raise ValueError("Scorer must return one score per text pair")
    scores = []
    for value in raw_scores:
        if isinstance(value, (bool, str, bytes)) or getattr(value, "ndim", 0) != 0:
            raise ValueError("Scorer must return finite scalar scores")
        try:
            score = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Scorer must return finite scalar scores") from exc
        if not isfinite(score):
            raise ValueError("Scorer must return finite scalar scores")
        scores.append(score)
    scored = []
    for slot, score in zip(slots, scores):
        row = rows[slot].model_copy(update={"cross_encoder_score": score})
        scored.append(row)
    scored.sort(key=lambda row: -row.cross_encoder_score)
    for slot, row in zip(slots, scored):
        rows[slot] = row
    return [row.model_copy(update={"rank": index})
            for index, row in enumerate(rows, 1)]
