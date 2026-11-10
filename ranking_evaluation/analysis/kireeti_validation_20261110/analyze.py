"""Recompute descriptive diagnostics from frozen frontend artifacts; no model inference."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'explanation_frontend/evaluation/validation/results'
rows = list(csv.DictReader((SOURCE / 'test_results.csv').open(encoding='utf-8-sig')))
names = ['oracle_gold_rationale', 'gold_papers_production_selector', 'retrieved_papers_production_selector']
groups = {name: {r['claim_id']: r for r in rows if r['condition'] == name} for name in names}
assert len(rows) == 676 and len(groups[names[2]]) == 300
positive_ids = set(groups[names[0]])
assert len(positive_ids) == 188 and positive_ids == set(groups[names[1]])
summary = {'scope': 'Artifact-only audit; no new neural inference or parameter tuning.', 'conditions': {}, 'paired_transitions': {}}
for name, group in groups.items():
    positive = [group[i] for i in positive_ids]
    counts = {key: sum(int(r[key]) for r in positive) for key in ['gold_documents', 'gold_documents_retrieved', 'complete_rationale_docs', 'gold_rationale_docs', 'partial_sentence_fragments', 'rejected_spans']}
    summary['conditions'][name] = {'all_count': len(group), 'all_correct': sum(r['correct'] == 'True' for r in group.values()), 'positive_count': len(positive), 'positive_correct': sum(r['correct'] == 'True' for r in positive), **counts}
for first, second in zip(names, names[1:]):
    summary['paired_transitions'][first + ' -> ' + second] = dict(Counter(('correct' if groups[first][i]['correct'] == 'True' else 'wrong') + ' -> ' + ('correct' if groups[second][i]['correct'] == 'True' else 'wrong') for i in positive_ids))
summary['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE / 'test_results.csv', SOURCE / 'error_catalogue.csv']}
cases = []
for p in sorted((SOURCE / 'replays').glob('claim-*.json')):
    cid = p.stem.removeprefix('claim-')
    snapshot = json.loads(p.read_text())['snapshot']
    cases.append({'claim_id': cid, 'result': groups[names[2]][cid], 'condition_results': {n: groups[n].get(cid) for n in names}, 'replay': str(p.relative_to(ROOT)), 'claim': snapshot['claim'], 'verification': snapshot['verification']})
(OUT / 'diagnostics.json').write_text(json.dumps(summary, indent=2) + '\n')
(OUT / 'replay_cases.json').write_text(json.dumps(cases, indent=2) + '\n')
print(json.dumps(summary, indent=2))
