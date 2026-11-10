"""Inspect source-aligned selection and context coverage without neural inference."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'data/scifact'
EXPECTED = {'corpus.jsonl': 'b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62', 'claims_dev.jsonl': '86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217'}
for name, digest in EXPECTED.items():
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest
corpus = {str(r['doc_id']): r for r in map(json.loads, (DATA / 'corpus.jsonl').open())}
claims = {str(r['id']): r for r in map(json.loads, (DATA / 'claims_dev.jsonl').open())}
results = []
for cid in ['137', '1163']:
    claim = claims[cid]
    replay = json.loads((ROOT / f'explanation_frontend/evaluation/validation/results/replays/claim-{cid}.json').read_text())
    verification = replay['snapshot']['verification']
    papers = []
    for pid, annotations in claim['evidence'].items():
        sentences = corpus[pid]['abstract']
        source = ' '.join(sentences)
        intervals = []
        offset = 0
        for sentence in sentences:
            intervals.append((offset + len(sentence) - len(sentence.lstrip()), offset + len(sentence.rstrip())))
            offset += len(sentence) + 1
        cuts = [0, *[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9])', source)], len(source)]
        segments = []
        for a, b in zip(cuts, cuts[1:]):
            raw = source[a:b]
            start, end = a + len(raw) - len(raw.lstrip()), a + len(raw.rstrip())
            if start < end:
                segments.append((start, end))
        selected = [p for p in verification['evidence_passages'] if p['paper_id'] == pid]
        chosen = []
        for passage in selected:
            matched = [i for i, (a, b) in enumerate(segments) if source[a:b] == passage['text']]
            assert len(matched) == 1, 'Expected uniquely source-aligned single-segment anchors'
            chosen.extend(matched)
        gold = sorted({i for ev in annotations for i in ev['sentences']})
        variants = []
        for radius in [0, 1]:
            expanded = sorted({j for i in chosen for j in range(max(0, i-radius), min(len(segments), i+radius+1))})
            covered = [i for i, (a, b) in enumerate(intervals) if any(segments[j][0] <= a and segments[j][1] >= b for j in expanded)]
            variants.append({'production_segment_context_radius': radius, 'selected_segment_count': len(expanded), 'covered_original_sentence_indices': covered, 'recovered_gold_indices': sorted(set(covered) & set(gold))})
        papers.append({'paper_id': pid, 'gold_sentence_indices': gold, 'original_sentences': sentences, 'production_segments': [{'index': i, 'start': a, 'end': b, 'text': source[a:b]} for i, (a,b) in enumerate(segments)], 'selected_segment_indices': chosen, 'selected_passages': selected, 'coverage_only_counterfactuals': variants})
    results.append({'claim_id': cid, 'claim': claim['claim'], 'papers': papers, 'per_passage_labels': verification['per_passage_labels']})
(OUT / 'selection_cases.json').write_text(json.dumps({'scope': 'Fixed archived anchors; source-segment neighbor expansion only. No reranking, tokenizer budget checks, NLI inference or new accuracy measurement.', 'source_sha256': EXPECTED, 'cases': results}, indent=2) + '\n')
for case in results:
    for paper in case['papers']:
        print(case['claim_id'], paper['paper_id'], 'gold', paper['gold_sentence_indices'], 'anchors', paper['selected_segment_indices'], 'variants', paper['coverage_only_counterfactuals'])
