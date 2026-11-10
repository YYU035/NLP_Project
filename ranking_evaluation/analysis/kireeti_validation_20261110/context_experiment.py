"""Fixed-anchor context replay with exact pinned verifier tokenization; no neural inference."""
import csv
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ranking_evaluation.semantic_evidence import SelectionConfig, select_scored_sentences
from ranking_evaluation.verifier_budget import enforce_verifier_budget, VERIFIER_CONTRACT
from explanation_frontend.evaluation.validation.scifact import sentence_recovery
from explanation_frontend.evaluation.validation.metrics import evidence_score
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'explanation_frontend/evaluation/validation/results'
DATA = ROOT / 'data/scifact'
EXPECTED = json.loads((OUT / 'selection_cases.json').read_text())['source_sha256']
for name, digest in EXPECTED.items():
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest
corpus = {str(r['doc_id']): r for r in map(json.loads, (DATA / 'corpus.jsonl').open())}
claims = {str(r['id']): r for r in map(json.loads, (DATA / 'claims_dev.jsonl').open())}
tokenizer_path = ROOT / '.cache/ranking/verifier-tokenizer'
tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
archived = {r['claim_id']: r for r in csv.DictReader((SOURCE / 'test_results.csv').open(encoding='utf-8-sig')) if r['condition'] == 'retrieved_papers_production_selector'}
rows = []
for replay_path in sorted((SOURCE / 'replays').glob('claim-*.json')):
    bundle = json.loads(replay_path.read_text())
    cid = replay_path.stem.removeprefix('claim-')
    claim = claims[cid]
    passages = bundle['snapshot']['verification']['evidence_passages']
    for radius in [0, 1]:
        started = perf_counter()
        recovered, partial = set(), set()
        details = []
        for pid in bundle['selected_paper_ids']:
            source = ' '.join(corpus[pid]['abstract'])
            cuts = [0, *[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9])', source)], len(source)]
            offsets = []
            for a,b in zip(cuts,cuts[1:]):
                raw = source[a:b]
                a,b = a+len(raw)-len(raw.lstrip()), a+len(raw.rstrip())
                if a < b: offsets.append((a,b))
            texts = [source[a:b] for a,b in offsets]
            scores = [-1.] * len(texts)
            anchors = [p for p in passages if p['paper_id'] == pid]
            for p in anchors:
                matches = [i for i,t in enumerate(texts) if t == p['text']]
                assert len(matches) == 1
                scores[matches[0]] = p['score']
            proposed = select_scored_sentences(pid, texts, scores, config=SelectionConfig(3,.5,radius))
            exact = []
            for s in proposed:
                a = offsets[s.sentence_indices[0]][0]
                b = offsets[s.sentence_indices[-1]][1]
                text = source[a:b]
                exact.append(replace(s, start=a, end=b, text=text, passage_id=f'{pid}:abstract:{a}:{b}:{hashlib.sha256(text.encode()).hexdigest()[:12]}'))
            decision = enforce_verifier_budget(claim['claim'], exact, tokenizer)
            if radius == 0:
                assert {s.text for s in decision.accepted} == {p['text'] for p in anchors}, 'Baseline replay mismatch'
            whole, fragment = sentence_recovery(corpus[pid]['abstract'], decision.accepted)
            recovered.update((pid,i) for i in whole)
            partial.update((pid,i) for i in fragment)
            details.append({'paper_id':pid, 'budget_checks':list(decision.checked), 'accepted_passages':[{'text':s.text,'start':s.start,'end':s.end,'passage_id':s.passage_id} for s in decision.accepted]})
        gold = {(pid,i) for pid,evs in claim['evidence'].items() for ev in evs for i in ev['sentences']}
        ev = evidence_score(len(recovered & gold), len(recovered-gold)+len(partial),len(gold-recovered))
        if radius == 0:
            assert ev == json.loads(archived[cid]['evidence']), 'Archived evidence score mismatch'
        complete = sum(any({(pid,i) for i in e['sentences']} <= recovered for e in evs) for pid,evs in claim['evidence'].items())
        rows.append({'claim_id':cid,'positive':bool(gold),'radius':radius,'evidence':ev,'complete_rationale_docs':complete,'gold_rationale_docs':len(claim['evidence']),'rejected':sum(not c['accepted'] for d in details for c in d['budget_checks']),'passages':sum(len(d['accepted_passages']) for d in details),'selection_and_tokenization_seconds':perf_counter()-started,'details':details})
summary = []
for radius in [0,1]:
    group = [r for r in rows if r['radius']==radius]
    positive = [r for r in group if r['positive']]
    summary.append({'radius':radius,'claims':len(group),'positive_claims':len(positive),'positive_evidence_micro':evidence_score(*(sum(r['evidence'][k] for r in positive) for k in ['tp','fp','fn'])),'complete_rationale_docs':sum(r['complete_rationale_docs'] for r in positive),'gold_rationale_docs':sum(r['gold_rationale_docs'] for r in positive),'rejected_spans_all_claims':sum(r['rejected'] for r in group),'accepted_passages_all_claims':sum(r['passages'] for r in group)})
manifest = {'protocol':'All 15 archived full-route replays, selected nonrandom diagnostic sample; fixed archived accepted anchors; unchanged top-five scope; radius 0 vs radius 1; overlap merging and whole-passage rejection at 512 pair tokens. No fallback after rejection.','limitations':['No encoder rerun; cannot recover anchors absent from archived accepted passages.','No NLI inference or new verdict accuracy/Macro-F1/latency measurement.','Not representative full-development or test results; no statistical improvement claim.','Timing includes selection and tokenization only, not model inference or end-to-end latency.'],'source_sha256':EXPECTED,'tokenizer_contract':VERIFIER_CONTRACT,'tokenizer_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tokenizer_path.iterdir() if p.is_file()},'summary':summary,'rows':rows}
(OUT / 'context_results.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
