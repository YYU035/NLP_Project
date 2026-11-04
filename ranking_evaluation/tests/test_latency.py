"""Test timing arithmetic with a deterministic clock, not speed assertions."""

import pytest
from pydantic import ValidationError

from ranking_evaluation.latency import LatencyRecord, measure_call


def test_measures_operation_and_keeps_unknown_stages():
    ticks = iter([10.0, 10.25])
    value, seconds = measure_call(lambda: ["result"], clock=lambda: next(ticks))
    assert value == ["result"]
    assert seconds == 0.25
    record = LatencyRecord(rrf_seconds=seconds)
    assert record.end_to_end_seconds is None
    assert record.bm25_seconds is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_rejects_invalid_durations(value):
    with pytest.raises(ValidationError):
        LatencyRecord(rrf_seconds=value)


def test_propagates_operation_failure():
    def fail():
        raise RuntimeError("operation failed")
    with pytest.raises(RuntimeError, match="operation failed"):
        measure_call(fail)


def test_rejects_backward_clock():
    ticks = iter([2, 1])
    with pytest.raises(ValueError, match="monotonic"):
        measure_call(lambda: None, clock=lambda: next(ticks))
