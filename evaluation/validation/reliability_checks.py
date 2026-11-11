"""Small actual-model repeatability and explanation checks; no accuracy selection."""
import json,time
import numpy as np
from evaluation.validation.freeze import ROOT, OUT
from evaluation.validation.retrieval import rank_dataset

def main():
    import torch
    torch.set_num_threads(6)
    from research_verifier.pipeline import local_encoder, local_verifier
    from research_verifier.interpretability.occlusion import explain_occlusion
    from evaluation.occlusion_qc import faithfulness
    corpus={str(d['doc_id']):d for d in map(json.loads,(ROOT/'outputs/xai/data/corpus.jsonl').read_text(encoding='utf-8').splitlines())}
    claims=list(map(json.loads,(ROOT/'outputs/xai/data/claims_dev.jsonl').read_text(encoding='utf-8').splitlines()))[:10]
    queries={str(c['id']):{'text':c['claim'],'candidate_ids':list(corpus)} for c in claims}
    saved=json.loads((OUT/'scifact_full/rankings.json').read_text(encoding='utf-8'))
    ranked=rank_dataset(corpus,queries,local_encoder(),stemmer='porter',model_key='production-MiniLM2a7-seq256',branch_depth=100)
    checks=[{'claim_id':q,'method':m,'passed':order[:20]==saved[m][q][:20]} for m,qq in ranked.items() for q,order in qq.items()]
    v=local_verifier();stress=json.loads((OUT/'stress.json').read_text(encoding='utf-8'));xai=[]
    for case in [stress['synthetic_nli_cases'][2],stress['synthetic_nli_cases'][4]]:
        a=explain_occlusion(v,case['claim'],case['evidence']);b=explain_occlusion(v,case['claim'],case['evidence'])
        difference=max(abs(x['attribution']-y['attribution']) for x,y in zip(a['words'],b['words']))
        bounds=all((case['evidence'] if w['segment']=='evidence' else case['claim'])[w['start']:w['end']]==w['text'] for w in a['words'])
        xai.append({'case_id':case['case_id'],'scope':'Synthetic predicted-class explanation; includes an actual wrong prediction. Not benchmark plausibility or correctness.',
            'predicted_target':a['target_label'],'model_prediction_correct':case['passed'],'word_count':len(a['words']),'coverage':a.get('coverage'),
            'max_repeat_attribution_difference':difference,'repeat_passed':difference<=1e-6,'offsets_passed':bounds,'faithfulness':faithfulness(v,case['claim'],case['evidence'],a,seed=20261110),
            'attribution':a})
    report={'retrieval_top20_repeated':checks,'xai':xai,'limitations':['Repeatability is checked in this CPU/software environment only.','Occlusion explains model behaviour, including errors; it does not validate the verdict.','MASK attribution and deletion diagnostics use different perturbations; neither proves causality or human plausibility.','Previously failed IG convergence checks remain failed; no IG success claim.']}
    (OUT/'reliability.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'retrieval_checks':len(checks),'retrieval_passed':sum(c['passed'] for c in checks),'xai_cases':len(xai),'xai_repeat_passed':sum(c['repeat_passed'] for c in xai),'xai_offsets_passed':sum(c['offsets_passed'] for c in xai)}),flush=True)
if __name__=='__main__':main()
