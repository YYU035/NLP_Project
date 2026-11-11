"""Additional peer-contract probes; never modify the owning teammate modules."""
import difflib
import json
from pathlib import Path
import subprocess

import numpy as np

from .._upstream.retrieval.retriever import _top_k
from .._upstream.claim_verifier.verifier import ClaimVerifier
from .._upstream.claim_verifier.schemas import EvidencePassage, PassageVerdict
from ..ranking import runtime

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "outputs" / "team_qc"
    output.mkdir(parents=True, exist_ok=True)
    scores = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
    expected = np.lexsort((np.arange(len(scores)), -scores))[:3].tolist()
    observed = _top_k(scores, 3).tolist()
    engine = object.__new__(ClaimVerifier)
    engine.checkpoint, engine.revision = 'contract-probe', 'synthetic'
    engine.max_length, engine.threshold, engine.seed = 512, .50, 42
    engine._classify_passage = lambda claim, p: PassageVerdict(
        passage_id=p.passage_id, paper_id=p.paper_id, label='SUPPORT', confidence=.9)
    wrong = EvidencePassage(passage_id='wrong', paper_id='outside-paper', section='abstract', text='Synthetic text.')
    response = engine.verify('synthetic claim', [{'paper_id': 'selected-paper', 'title': 'Synthetic'}],
                             {'selected-paper': [wrong]})
    result = {"naila_tie_cutoff": {"expected_index_order": expected, "actual_index_order": observed,
        "documented_rule_satisfied": expected == observed},
        "person4_foreign_passage": {"rejected": False,
            "paper_verdict_id": response.per_paper_verdicts[0].paper_id,
            "passage_paper_id": response.per_passage_labels[0].paper_id,
            "frontend_protection": "RunSnapshot validation rejects the mismatched lineage"}}
    patches = ROOT / 'qc_suggestions'
    patches.mkdir(exist_ok=True)
    old = (ROOT / '_upstream' / 'retrieval' / 'retriever.py').read_text(encoding='utf-8')
    new = old.replace('idx = np.argpartition(-scores, k - 1)[:k]\n    return idx[np.lexsort((idx, -scores[idx]))]',
                      'return np.lexsort((np.arange(len(scores)), -scores))[:k]')
    (patches / 'naila-stable-cutoff.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),
        new.splitlines(True), fromfile='a/retrieval/retriever.py', tofile='b/retrieval/retriever.py')), encoding='utf-8')
    # Run both ranking owners against identical real fixed-corpus branch lists.
    source = ROOT / 'outputs' / 'development_qc' / 'retrieval_records.json'
    if source.exists():
        rows = json.loads(source.read_text(encoding='utf-8'))
        Input, _, rank_claim = runtime()
        corpus = {str(x['doc_id']): x for x in map(json.loads, subprocess.check_output([
            'git', 'show', '1ac06b675e3d0719a63cb8efe98eda4f2c5fd576:claim_verifier/data/corpus.jsonl'],
            cwd=ROOT.parent).decode().splitlines())}
        claims = {str(x['id']): x for x in map(json.loads, subprocess.check_output([
            'git', 'show', '1ac06b675e3d0719a63cb8efe98eda4f2c5fd576:claim_verifier/data/claims_dev.jsonl'],
            cwd=ROOT.parent).decode().splitlines())}
        agreement = []
        for cid in sorted({x['claim_id'] for x in rows}):
            methods = {x['method']: x for x in rows if x['claim_id'] == cid}
            ids = sorted(set(methods['bm25']['ranked_ids']) | set(methods['sbert']['ranked_ids']))
            entries = [{'identity': pid, 'identity_aliases': [pid], 'original_claim': claims[cid]['claim'],
                        'paper': {'title': corpus[pid]['title'], 'abstract': ' '.join(corpus[pid]['abstract']),
                                  'source': 'scifact', 'source_id': pid}, 'provenance': [], 'source_records': []}
                       for pid in ids]
            inputs = Input.model_validate({'claim_id': cid,
                'candidates': {'original_claim': claims[cid]['claim'], 'top_k': 100, 'papers': entries},
                'bm25_ranking': methods['bm25']['ranked_ids'], 'sbert_ranking': methods['sbert']['ranked_ids']})
            expected_ids = methods['rrf']['ranked_ids']
            actual_ids = [p.entry.identity for p in rank_claim(inputs).papers][:100]
            agreement.append(actual_ids == expected_ids)
        result['rrf_cross_owner_real_data'] = {'claims': len(agreement), 'identical_top100': sum(agreement)}
    (output / 'contract-probes.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
