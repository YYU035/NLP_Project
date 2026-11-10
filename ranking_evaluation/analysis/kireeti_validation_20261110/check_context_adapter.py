"""Check the reusable adapter against every actual evaluated baseline/context trace."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from ranking_evaluation.contextual_evidence import BoundedAbstractSelector
from ranking_evaluation.evidence_selection import EvidenceSpan
from ranking_evaluation.semantic_evidence import SelectionConfig
from ranking_evaluation.verifier_budget import enforce_verifier_budget
from transformers import AutoTokenizer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    here=Path(__file__).resolve().parent
    frozen=json.loads((here/'relevance-gate-results/train-frozen-choice.json').read_text())
    assert frozen['selected_policy']=='bounded_abstract'
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'.cache/ranking/verifier-model'),local_files_only=True)
    docs={str(r['doc_id']):r for r in map(json.loads,(ROOT/'data/scifact/corpus.jsonl').open())}
    claims={str(r['id']):r for split in ['train','dev'] for r in map(json.loads,(ROOT/f'data/scifact/claims_{split}.jsonl').open())}
    checks=0
    for split in ['train','dev']:
        for path in sorted((args.run/f'{split}-traces').glob('*-baseline.json')):
            cid=path.stem.removesuffix('-baseline');base=json.loads(path.read_text());context=json.loads(path.with_name(f'{cid}-bounded_abstract.json').read_text())
            source_passages=base['response']['evidence_passages'];expected_passages=context['response']['evidence_passages']
            for pid in base['scope_paper_ids']:
                abstract=' '.join(docs[pid]['abstract'])
                spans=[]
                for p in source_passages:
                    if p['paper_id']!=pid:continue
                    parts=p['passage_id'].split(':');a,b=int(parts[-3]),int(parts[-2])
                    spans.append(EvidenceSpan(pid,(),a,b,p['text'],p['score'],p['passage_id']))
                class Base:
                    config=SelectionConfig(3,.5,0)
                    def select_abstract(self,*args):return spans
                wrapped=BoundedAbstractSelector(Base(),tokenizer,min_paper_score=frozen['paper_gate'])
                got=enforce_verifier_budget(claims[cid]['claim'],wrapped.select_abstract(claims[cid]['claim'],pid,abstract),tokenizer).accepted
                expected=[p for p in expected_passages if p['paper_id']==pid]
                if max((p['score'] for p in expected),default=-1.)<frozen['paper_gate']:expected=[]
                assert [(s.passage_id,s.text,s.score) for s in got]==[(p['passage_id'],p['text'],p['score']) for p in expected], (cid,pid)
                checks+=1
    result={'source_run':str(args.run),'policy':'bounded_abstract','gate':frozen['paper_gate'],'paper_scope_checks_passed':checks,'scope':'Adapter accepted passages match evaluated policy and paper gate; base proposals reconstructed from actual accepted baseline traces. Original rejected proposals are not reconstructable from these traces.'}
    (here/'relevance-gate-results/adapter-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
