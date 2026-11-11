"""Monotonic wall-clock measurements; no inferred end-to-end timings."""

from collections.abc import Callable
from time import perf_counter
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class LatencyRecord(BaseModel):
    """Unknown stages remain None; concurrent branches must not be summed."""

    model_config = ConfigDict(extra="forbid")

    search_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    bm25_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    sbert_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    rrf_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    rerank_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_to_end_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)


def measure_call(
    operation: Callable[[], T], *, clock: Callable[[], float] = perf_counter,
) -> tuple[T, float]:
    """Time one synchronous operation including its validation/serialization.

    Exceptions propagate. For async/GPU work, callers must await/synchronize
    completion within an appropriate timer; this helper is for CPU sync calls.
    """
    start = clock()
    result = operation()
    elapsed = clock() - start
    if elapsed < 0:
        raise ValueError("Clock must be monotonic")
    return result, elapsed
