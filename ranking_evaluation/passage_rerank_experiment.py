"""Matched top-1 passage-budget reranker on/off development ablation."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from ranking_evaluation.evidence_selection import EvidenceSpan
from ranking_evaluation.evaluate_evidence import score_selection,summarize
from ranking_evaluation.frontend_bridge import load_verifier_tokenizer
from ranking_evaluation.passage_reranker import rerank_passages
from ranking_evaluation.reranker import LocalCrossEncoder
from ranking_evaluation.scifact import load_scifact,sha256
from ranking_evaluation.verifier_budget import enforce_verifier_budget


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['corpus','claims','predictions','model-path','verifier-tokenizer','output-dir']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    docs,claims=load_scifact(a.corpus,a.claims,split='dev')
    manifest=json.loads((a.model_path/'download_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha256(a.model_path/name)!=expected:raise ValueError('Reranker model hash mismatch')
    scorer=LocalCrossEncoder(a.model_path,device='cpu',max_length=512,batch_size=16)
    verifier_tokenizer=load_verifier_tokenizer(str(a.verifier_tokenizer.resolve()))
    groups=defaultdict(list);cache={};excluded=0
    with a.predictions.open() as source,(a.output_dir/'per-claim.jsonl').open('x') as out:
        for line in source:
            row=json.loads(line)
            if row['policy']!='semantic_sentence' or not (row['ranking_method']=='oracle_gold_papers' or row['ranking_method'] in ['rrf','rrf_ce_top20'] and row['paper_depth']==5):continue
            cid=row['claim_id'];claim=claims[cid]
            per_paper=defaultdict(list)
            for p in row['passages']:per_paper[p['paper_id']].append(EvidenceSpan(**{**p,'sentence_indices':tuple(p['sentence_indices'])}))
            predictions={'reranker_off':set(),'reranker_on':set()};chosen={'reranker_off':[],'reranker_on':[]}
            for doc,spans in per_paper.items():
                key=(cid,doc)
                if key not in cache:
                    eligible=list(enforce_verifier_budget(claim.claim,spans,verifier_tokenizer).accepted)
                    checked=[]
                    for span in eligible:
                        if ' '.join(docs[doc].abstract)[span.start:span.end]!=span.text:raise ValueError('Source mismatch')
                        if len(scorer.model.tokenizer.encode(claim.claim,span.text,add_special_tokens=True,truncation=False))<=512:checked.append(span)
                        else:excluded+=1
                    cache[key]={'reranker_off':checked[:1],'reranker_on':rerank_passages(claim.claim,checked,scorer,top_k=1)}
                for mode,selected in cache[key].items():
                    for span in selected:
                        predictions[mode].update((doc,i) for i in span.sentence_indices);chosen[mode].append(asdict(span))
            for mode,predicted in predictions.items():
                result={'claim_id':cid,'mode':mode,'ranking_method':row['ranking_method'],'paper_depth':row['paper_depth'],
                    'passages':chosen[mode],**score_selection(predicted,claim.evidence)}
                groups[(mode,row['ranking_method'],row['paper_depth'])].append(result)
                out.write(json.dumps(result,ensure_ascii=False)+'\n')
    summaries=[{'mode':mode,'ranking_method':method,'paper_depth':depth,**summarize(rows)} for (mode,method,depth),rows in groups.items()]
    (a.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (a.output_dir/'manifest.json').write_text(json.dumps({'status':'complete','protocol':'fixed semantic candidate pool, identical top1 per-paper budget; reranker on/off, no dev tuning',
        'score_semantics':'reranker_on scores are raw relevance logits, not cosine or truth probabilities',
        'model':manifest,'predictions_sha256':sha256(a.predictions),'additional_ce_token_exclusions':excluded,
        'limits':'both actual NLI pair budget and Cross-Encoder token budget checked without truncation'},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
