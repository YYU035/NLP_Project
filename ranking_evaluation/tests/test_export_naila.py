"""Integration checks for stable cutoff and complete, gold-independent exports."""

from types import SimpleNamespace

import pytest

from ranking_evaluation.export_naila import export_rankings, stable_top_k, validate_source


def test_cutoff_ties_include_earliest_corpus_positions():
    assert stable_top_k([0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [str(i) for i in range(10)], 3) == ['4', '0', '1']
    with pytest.raises(ValueError, match='finite'):
        stable_top_k([float('nan')], ['a'], 1)


def test_export_preserves_nei_and_never_supplies_gold_to_scorers():
    class Scorer:
        paper_ids = ['a', 'b']
        def __init__(self):
            self.queries = []
        def score_all(self, query):
            self.queries.append(query)
            return [0, 1]

    bm25, sbert = Scorer(), Scorer()
    claims = {'1': SimpleNamespace(claim='positive', evidence={'a': ['gold']}),
              '2': SimpleNamespace(claim='no evidence', evidence={})}
    records = export_rankings(papers=[SimpleNamespace(paper_id=p) for p in ['a', 'b']],
                              claims=claims, bm25=bm25, sbert=sbert, k=2)
    assert [r.claim_id for r in records] == ['1', '2']
    assert all(r.bm25_ranking == ['b', 'a'] and r.sbert_ranking == ['b', 'a'] for r in records)
    assert bm25.queries == sbert.queries == ['positive', 'positive', 'no evidence']


def test_source_hash_mismatch_rejected(tmp_path):
    import json
    (tmp_path / 'source_manifest.json').write_text(json.dumps({
        'commit': 'abc', 'files_sha256': {'module.py': 'incorrect'},
    }))
    (tmp_path / 'module.py').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        validate_source(tmp_path)
