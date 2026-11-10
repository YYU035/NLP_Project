"""Adaptive composite IG on the same embedding path, using Captum quadrature.
Owner tokenizer, baseline, target and word/sentence aggregation stay unchanged.
"""
import time
import torch
from captum.attr import IntegratedGradients


def adaptive_run(explainer, premise, claim, target=None, max_evaluations=512):
    enc=explainer._encode(premise,claim)
    if enc['truncated']: raise ValueError('Adaptive attribution refuses truncated inputs')
    t=enc['tensors']; ids=t['input_ids']
    baseline_ids=ids.clone()
    for index,segment in enumerate(enc['seq_ids']):
        if segment is not None: baseline_ids[0,index]=explainer.baseline_token_id
    model=explainer.model
    def forward(embeddings):
        # Captum batches alpha points; expand only the unchanged original masks.
        kwargs={k:v.expand(embeddings.shape[0],-1) for k,v in t.items() if k!='input_ids'}
        return model(inputs_embeds=embeddings,**kwargs).logits.float().softmax(-1)
    with torch.no_grad():
        original=model.get_input_embeddings()(ids).detach()
        baseline=model.get_input_embeddings()(baseline_ids).detach()
        probs=explainer._forward(ids,t['attention_mask'],t.get('token_type_ids'))[0]
        parity=float((forward(original)[0]-probs).abs().max())
        base_probs=forward(baseline)[0]
    if parity>1e-6: raise ValueError('Embedding-path probability differs from original verifier')
    target_idx,pred_label=explainer._target_index(probs,target)
    p,p_base=float(probs[target_idx]),float(base_probs[target_idx])
    direction=original-baseline
    ig=IntegratedGradients(forward)
    evaluations=0
    trace=[]
    def panel(a,b):
        nonlocal evaluations
        low=baseline+a*direction; high=baseline+b*direction
        values=[]
        for order in (8,16):
            attrs=ig.attribute(high,baselines=low,target=target_idx,n_steps=order,internal_batch_size=1)
            values.append(attrs.sum(-1)[0].detach().double().cpu())
            evaluations+=order
        error=float((values[1]-values[0]).abs().sum())
        return {'a':a,'b':b,'value':values[1],'error':error}
    if max_evaluations<24: raise ValueError('Adaptive IG needs at least24 evaluations')
    panels=[panel(0.,1.)]
    while True:
        total=torch.stack([p['value'] for p in panels]).sum(0)
        delta=float(total.sum())-(p-p_base)
        estimated_error=sum(p['error'] for p in panels)
        local_limit=max(.0001,.01*float(total.abs().sum()))
        passed=abs(delta)<=.001 and abs(delta)/max(abs(p-p_base),1e-8)<=.05 and estimated_error<=local_limit
        trace.append({'evaluations':evaluations,'panels':len(panels),'absolute_error':abs(delta),
                      'relative_error':abs(delta)/max(abs(p-p_base),1e-8),
                      'estimated_token_l1_error':estimated_error,'local_error_limit':local_limit,'passed':passed})
        if passed or evaluations+48>max_evaluations:break
        worst=max(panels,key=lambda item:item['error']);panels.remove(worst)
        mid=(worst['a']+worst['b'])/2
        panels.extend([panel(worst['a'],mid),panel(mid,worst['b'])])
    run={'enc':enc,'target_idx':target_idx,'pred_label':pred_label,'token_attr':total.tolist(),
         'p':p,'p_base':p_base,'delta':delta}
    diagnostics={'method':'Adaptive composite IntegratedGradients (Captum), embedding output path',
                 'evaluations':evaluations,'budget':max_evaluations,'quadrature_orders':[8,16],
                 'input_probability_parity_error':parity,'local_quadrature_passed':passed,
                 'trace':trace,'error_estimate_note':'Embedded8/16-order token-L1 discrepancy; diagnostic estimate, not a rigorous bound.'}
    return run,diagnostics
