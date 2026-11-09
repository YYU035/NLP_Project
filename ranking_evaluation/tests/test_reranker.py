"""Offline scorer injection tests; these do not validate pretrained inference."""

import pytest

from ranking_evaluation.examples.mock_input import build_mock_input
from ranking_evaluation.reranker import LocalCrossEncoder, rerank
from ranking_evaluation.rrf import reciprocal_rank_fusion


class FakeScorer:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = []

    def score_pairs(self, pairs):
        self.pairs = pairs
        return self.scores


def sample():
    inputs = build_mock_input()
    return inputs.candidates.original_claim, reciprocal_rank_fusion(inputs)


def test_reranks_prefix_preserves_tail_metadata_and_input():
    claim, rows = sample()
    before = [row.model_dump() for row in rows]
    scorer = FakeScorer([-2, 3])
    result = rerank(rows, claim=claim, scorer=scorer, top_n=2)
    assert [row.entry.identity for row in result] == [
        "synthetic:B", "synthetic:A", "synthetic:C",
    ]
    assert [row.rank for row in result] == [1, 2, 3]
    assert [row.cross_encoder_score for row in result] == [3, -2, None]
    assert result[0].entry == rows[1].entry
    assert result[0].rrf_score == rows[1].rrf_score
    assert scorer.pairs == [(claim, row.entry.paper.title) for row in rows[:2]]
    assert [row.model_dump() for row in rows] == before


def test_missing_text_keeps_slot_and_abstract_only_is_scored():
    claim, rows = sample()
    rows[1].entry.paper.title = None
    rows[0].entry.paper.title = None
    rows[0].entry.paper.abstract = " Abstract only "
    scorer = FakeScorer([0, 1])
    result = rerank(rows, claim=claim, scorer=scorer)
    assert [row.entry.identity for row in result] == [
        "synthetic:C", "synthetic:B", "synthetic:A",
    ]
    assert scorer.pairs[0] == (claim, "Abstract only")
    assert result[1].cross_encoder_score is None


def test_title_and_abstract_concatenation_and_stable_ties():
    claim, rows = sample()
    rows[0].entry.paper.abstract = "Abstract"
    scorer = FakeScorer([1, 1, 1])
    assert rerank(rows, claim=claim, scorer=scorer) == [
        row.model_copy(update={"cross_encoder_score": 1}) for row in rows
    ]
    assert scorer.pairs[0][1] == rows[0].entry.paper.title + "\n\nAbstract"


def test_bypass_empty_and_unscoreable_inputs():
    claim, rows = sample()
    assert rerank(rows, claim=claim, enabled=False) == rows
    assert rerank(rows, claim=claim, top_n=0) == rows
    assert rerank([], claim=claim) == []
    for row in rows:
        row.entry.paper.title = None
    scorer = FakeScorer([])
    assert rerank(rows, claim=claim, scorer=scorer) == rows
    assert scorer.pairs == []


@pytest.mark.parametrize("scores", [[1], [1, 2, float("nan")],
                                     [1, 2, float("inf")], [[1], [2], [3]],
                                     [True, 1, 2]])
def test_bad_scores_raise(scores):
    claim, rows = sample()
    with pytest.raises(ValueError):
        rerank(rows, claim=claim, scorer=FakeScorer(scores))


def test_invalid_inputs_and_missing_scorer():
    claim, rows = sample()
    with pytest.raises(ValueError, match="explicit scorer"):
        rerank(rows, claim=claim)
    with pytest.raises(ValueError, match="claim"):
        rerank(rows, claim="wrong", enabled=False)
    for top_n in (-1, True, 1.5):
        with pytest.raises(ValueError):
            rerank(rows, claim=claim, top_n=top_n)
    rows[0].rank = 2
    with pytest.raises(ValueError, match="consecutive"):
        rerank(rows, claim=claim, enabled=False)


def test_failure_propagates_and_stale_scores_cleared():
    claim, rows = sample()
    class Broken:
        def score_pairs(self, pairs):
            raise RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        rerank(rows, claim=claim, scorer=Broken())
    rows[0].cross_encoder_score = 100
    assert rerank(rows, claim=claim, enabled=False)[0].cross_encoder_score is None


def test_loader_rejects_nonlocal_path_without_importing_model(tmp_path):
    with pytest.raises(ValueError, match="local model directory"):
        LocalCrossEncoder(tmp_path / "missing")
    with pytest.raises(ValueError, match="batch_size"):
        LocalCrossEncoder(tmp_path, batch_size=0)


def test_local_adapter_passes_offline_cpu_and_batch_settings(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = {}
    class StubModel:
        def __init__(self, path, **kwargs):
            calls.update(path=path, init=kwargs)

        def predict(self, pairs, **kwargs):
            calls.update(pairs=pairs, predict=kwargs)
            return [0.5]

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(CrossEncoder=StubModel))
    adapter = LocalCrossEncoder(tmp_path, max_length=256, batch_size=4)
    assert adapter.score_pairs([("claim", "text")]) == [0.5]
    assert calls["init"] == {
        "device": "cpu", "max_length": 256,
        "local_files_only": True, "trust_remote_code": False,
    }
    assert calls["predict"]["batch_size"] == 4


def test_local_adapter_normalizes_only_single_label_dimension():
    numpy = pytest.importorskip("numpy")
    from types import SimpleNamespace

    adapter = LocalCrossEncoder.__new__(LocalCrossEncoder)
    adapter.batch_size = 2
    adapter.model = SimpleNamespace(predict=lambda *args, **kwargs:
                                    numpy.array([[1.0], [2.0]]))
    assert adapter.score_pairs([("c", "a"), ("c", "b")]).tolist() == [1, 2]
    adapter.model = SimpleNamespace(predict=lambda *args, **kwargs:
                                    numpy.array([[1.0, 2.0]]))
    assert adapter.score_pairs([("c", "a")]).shape == (1, 2)


def test_cuda_adapter_synchronizes_before_and_after_inference(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    events = []
    class StubModel:
        def __init__(self, path, **kwargs):
            assert kwargs['device'] == 'cuda'
        def predict(self, pairs, **kwargs):
            events.append('predict')
            return [0.5]
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(CrossEncoder=StubModel))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda device: events.append(('sync', device)))))
    adapter = LocalCrossEncoder(tmp_path, device='cuda')
    assert adapter.score_pairs([('claim', 'text')]) == [0.5]
    assert events == [('sync', 'cuda'), 'predict', ('sync', 'cuda')]


def test_mps_adapter_synchronizes_before_and_after_inference(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    events = []
    class StubModel:
        def __init__(self, path, **kwargs):
            assert kwargs['device'] == 'mps'
        def predict(self, pairs, **kwargs):
            events.append('predict')
            return [0.5]
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(CrossEncoder=StubModel))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(mps=SimpleNamespace(
        synchronize=lambda: events.append('sync'))))
    adapter = LocalCrossEncoder(tmp_path, device='mps')
    assert adapter.score_pairs([('claim', 'text')]) == [0.5]
    assert events == ['sync', 'predict', 'sync']
