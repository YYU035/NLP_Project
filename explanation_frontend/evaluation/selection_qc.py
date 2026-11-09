"""Strict sentence-coverage diagnostic for the provisional selector, not official SciFact scoring."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sentence_coverage(sentences, intervals):
    complete, partial = [], []
    offset = 0
    for index, text in enumerate(sentences):
        positions = [offset + i for i, char in enumerate(text) if not char.isspace()]
        covered = [any(a <= p < b for a, b in intervals) for p in positions]
        if covered and all(covered):
            complete.append(index)
        elif any(covered):
            partial.append(index)
        offset += len(text) + 1
    return complete, partial


def main():
    output = ROOT / 'outputs' / 'development_qc'
    manifest = json.loads((output / 'frozen_manifest.json').read_text(encoding='utf-8'))
    corpus = {str(p['doc_id']): p for p in map(json.loads, subprocess.check_output(['git', 'show',
        '1ac06b675e3d0719a63cb8efe98eda4f2c5fd576:claim_verifier/data/corpus.jsonl'],
        cwd=ROOT.parent).decode().splitlines())}
    records = []
    for index, case in enumerate(manifest['cases']):
        bundle = json.loads((output / f'case-{index:02d}.json').read_text(encoding='utf-8'))
        sentences = corpus[case['paper_id']]['abstract']
        body = ' '.join(sentences)
        intervals = []
        for passage in bundle['snapshot']['verification']['evidence_passages']:
            match = re.search(r':abstract:(\d+):(\d+):[0-9a-f]{12}$', passage['passage_id'])
            if not match:
                raise ValueError('No source offsets')
            start, end = map(int, match.groups())
            if body[start:end] != passage['text']:
                raise ValueError('Source text mismatch')
            intervals.append((start, end))
        complete, partial = sentence_coverage(sentences, intervals)
        selected, gold = set(complete), set(case['sentence_indices'])
        tp = len(selected & gold)
        precision = tp / len(selected) if selected else 0
        recall = tp / len(gold)
        records.append({'claim_id': case['claim_id'], 'paper_id': case['paper_id'],
            'gold_indices': sorted(gold), 'fully_covered_indices': complete, 'partially_covered_indices': partial,
            'tp': tp, 'selected': len(selected), 'gold': len(gold),
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0})
    tp, predicted, gold = [sum(row[key] for row in records) for key in ['tp', 'selected', 'gold']]
    p, r = tp / predicted, tp / gold
    summary = {'cases': len(records), 'selected_sentences': predicted, 'gold_sentences': gold,
        'matched_sentences': tp, 'micro_precision': p, 'micro_recall': r, 'micro_f1': 2*p*r/(p+r) if p+r else 0,
        'macro_f1': sum(row['f1'] for row in records) / len(records),
        'cases_with_partial_sentence': sum(bool(row['partially_covered_indices']) for row in records),
        'source_span_checks_passed': len(records),
        'metric': 'Strict complete coverage of original SciFact sentence non-whitespace characters',
        'limitations': ['Small label-stratified development sample; oracle document supplied',
            'Only the chosen annotated rationale is counted; other valid rationales can be penalized',
            'Context neighbours count as false positives against that rationale',
            'Custom diagnostic, not the official SciFact scorer or full-pipeline evidence F1']}
    (output / 'selection_records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (output / 'selection_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
