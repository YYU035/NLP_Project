"""Check cluster resampling and cached-prefix replay against known examples."""

import pytest

from ranking_evaluation.comprehensive_evaluation import cluster_intervals, replay_prefix


def test_cluster_bootstrap_constant_metric_and_paired_difference():
    np = pytest.importorskip('numpy')
    point, samples, groups = cluster_intervals([[0.6, 0.8]] * 5,
        ['paper-a', 'paper-a', 'paper-b', 'paper-c', 'paper-c'], repeats=200, seed=7)
    assert groups == 3
    assert point.tolist() == pytest.approx([0.6, 0.8])
    assert np.allclose(samples[:, 1] - samples[:, 0], 0.2)
    assert np.allclose(samples[:, 0], 0.6)


def test_prefix_replay_preserves_tail_and_rejects_unscored_extension():
    before = ['a', 'b', 'c', 'd', 'e']
    cached = ['d', 'b', 'a', 'c', 'e']
    assert replay_prefix(before, cached, 4, 2) == ['b', 'a', 'c', 'd', 'e']
    assert replay_prefix(before, cached, 4, 0) == before
    with pytest.raises(ValueError, match='Invalid cached'):
        replay_prefix(before, cached, 4, 5)
    with pytest.raises(ValueError, match='Invalid cached'):
        replay_prefix(before, ['d', 'b', 'e', 'c', 'a'], 4, 2)


def test_bootstrap_rejects_invalid_observations():
    pytest.importorskip('numpy')
    with pytest.raises(ValueError, match='finite'):
        cluster_intervals([[float('nan')]], ['paper'], repeats=200)
