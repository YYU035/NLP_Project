"""Frozen four-case word-occlusion diagnostics with matched word-deletion controls."""
import json,random,statistics,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def faithfulness(v,claim,text,result,seed=42):
    import torch
    enc=v.tokenizer(text,claim,return_tensors='pt');ids=enc['input_ids'][0].tolist()
    tti=enc.get('token_type_ids');tti=tti[0].tolist() if tti is not None else None
    words=[w for w in result['words'] if w['segment']=='evidence']
    ranked=sorted(words,key=lambda w:w['attribution'],reverse=True)
    target=next(i for i,l in v.label_mapping.items() if l.value==result['target_label'])
    rng=random.Random(seed);ratios=[.01,.05,.1,.2,.5];variants=[];keys=[]
    all_idx=set(range(len(ids)));candidates={i for w in words for i in w['token_indices']}
    for ratio in ratios:
        k=max(1,round(len(words)*ratio))
        groups=[('occlusion',ranked[:k])]+[('random',rng.sample(words,k)) for _ in range(10)]
        for method,group in groups:
            chosen={i for w in group for i in w['token_indices']}
            variants.extend([sorted(all_idx-chosen),sorted((all_idx-candidates)|chosen)])
            keys.extend([(ratio,method,'comprehensiveness'),(ratio,method,'sufficiency')])
    probabilities=[]
    with torch.no_grad():
        for begin in range(0,len(variants),4):
            chunk=variants[begin:begin+4];length=max(map(len,chunk))
            tensors={'input_ids':torch.tensor([[ids[i] for i in row]+[v.tokenizer.pad_token_id]*(length-len(row)) for row in chunk],device=v.device),
                     'attention_mask':torch.tensor([[1]*len(row)+[0]*(length-len(row)) for row in chunk],device=v.device)}
            if tti is not None:tensors['token_type_ids']=torch.tensor([[tti[i] for i in row]+[0]*(length-len(row)) for row in chunk],device=v.device)
            probabilities.extend(float(p) for p in v.model(**tensors).logits.float().softmax(-1)[:,target])
    metrics={}
    for key,p in zip(keys,probabilities):metrics.setdefault(key,[]).append(result['target_probability']-p)
    report={}
    for method in ('occlusion','random'):
        for metric in ('comprehensiveness','sufficiency'):
            report[method+'_'+metric]=statistics.mean(statistics.mean(metrics[(r,method,metric)]) for r in ratios)
    report['unit']='whole evidence words; matched word counts, all subwords deleted together; claim/special tokens retained'
    report['random_trials']=10
    return report


def main():
    from research_verifier.pipeline import local_verifier
    from research_verifier.interpretability.occlusion import explain_occlusion
    from research_verifier.interpretability.metrics import build_examples, average_precision, random_ap
    from research_verifier.verification.schemas import PassageLabel
    from evaluation.xai_qc import sample_examples
    v=local_verifier();examples=build_examples(str(ROOT/'outputs/xai/data'))
    selected,_=sample_examples(examples,v.tokenizer,4,192,42)
    rows=[];directory=ROOT/'outputs/release/occlusion';directory.mkdir(parents=True,exist_ok=True)
    for index,ex in enumerate(selected):
        t=time.perf_counter();text=' '.join(ex['sentences'])
        r=explain_occlusion(v,ex['claim'],text,target=PassageLabel(ex['gold_label']),max_words=512)
        if not r['coverage']['complete']:raise ValueError('Incomplete attribution coverage')
        scores=[0.]*len(ex['sentences']);start=0
        for i,sentence in enumerate(ex['sentences']):
            end=start+len(sentence)
            scores[i]=sum(w['attribution'] for w in r['words'] if w['segment']=='evidence' and start<=w['start']<end)
            start=end+1
        relevant=set(ex['rationale']);rng=random.Random(42+index)
        row={'claim_id':ex['claim_id'],'doc_id':ex['doc_id'],'gold_label':ex['gold_label'],'predicted_label':r['predicted_label'],
             'ap':average_precision(scores,relevant),'ap_random':random_ap(len(scores),relevant,rng,trials=50),
             'top1_hit':max(range(len(scores)),key=lambda i:scores[i]) in relevant,'top1_random':len(relevant)/len(scores),
             'faithfulness':faithfulness(v,ex['claim'],text,r,seed=42+index),'seconds':time.perf_counter()-t}
        rows.append(row);(directory/f'case-{ex["claim_id"]}.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    summary={'date':'2026-11-09','method':'Word-level MASK occlusion, not IG','checkpoint':v.checkpoint,'revision':v.revision,
       'protocol':'Same seeded balanced4 SciFact full-abstract pairs as prior IG pilot; max192 tokens, gold target, complete word coverage. Exploratory, not held-out. Sentence AP uses signed word-score sums;50 random sentence trials. ERASER-style faithfulness uses word deletion and10 matched random trials, not official token-level parity.',
       'n':len(rows),'mean_ap':statistics.mean(r['ap'] for r in rows),'mean_ap_random':statistics.mean(r['ap_random'] for r in rows),
       'top1_hit_rate':statistics.mean(r['top1_hit'] for r in rows),'top1_random':statistics.mean(r['top1_random'] for r in rows),
       'faithfulness':{key:statistics.mean(r['faithfulness'][key] for r in rows) for key in ('occlusion_comprehensiveness','random_comprehensiveness','occlusion_sufficiency','random_sufficiency')},
       'cases':rows,'limitations':['Four short dev examples, not representative or a significance test.','Gold-target scores do not explain a different predicted class.','MASK/deletion perturbations can be unnatural; interactions are not additive.','Conditional model behaviour, not scientific truth; no IG completeness claim.']}
    summary['limitations'].append('Four claim-document pairs from three unique documents; two claims share one abstract. No independent-sample significance claim.')
    summary['by_raw_prediction_correctness']={}
    for correct in (True,False):
        group=[r for r in rows if (r['predicted_label']==r['gold_label'])==correct]
        if group:
            summary['by_raw_prediction_correctness']['correct' if correct else 'incorrect']={
                'n':len(group),'mean_ap':statistics.mean(r['ap'] for r in group),
                'mean_ap_random':statistics.mean(r['ap_random'] for r in group),
                'faithfulness':{k:statistics.mean(r['faithfulness'][k] for r in group) for k in summary['faithfulness']}}
    (ROOT/'evaluation/occlusion-results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return summary
if __name__=='__main__':main()
