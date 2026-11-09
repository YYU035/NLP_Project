"""Finite-difference audit of the actual model embedding path on a synthetic pair."""
import json
from pathlib import Path
import time


def main():
    import torch
    from ..pipeline import local_verifier
    v=local_verifier(); text='Plants require water to survive.'
    enc=v.tokenizer(text,text,return_tensors='pt')
    model=v.model; model.eval()
    output=[]
    def probabilities(embeddings):
        return model(inputs_embeds=embeddings,**{k:t for k,t in enc.items() if k!='input_ids'}).logits.float().softmax(-1)
    with torch.no_grad():
        real=model.get_input_embeddings()(enc['input_ids']).detach()
        direct=model(**enc).logits.float().softmax(-1)
        embedded=probabilities(real)
        target=int(direct.argmax())
        parity=float((direct-embedded).abs().max())
    for name in ('pad','mask'):
        baseline_ids=enc['input_ids'].clone()
        special=v.tokenizer.get_special_tokens_mask(baseline_ids[0].tolist(),already_has_special_tokens=True)
        for index,is_special in enumerate(special):
            if not is_special:baseline_ids[0,index]=getattr(v.tokenizer,name+'_token_id')
        with torch.no_grad():base=model.get_input_embeddings()(baseline_ids).detach()
        direction=real-base
        for alpha in (.01,.1,.25,.5,.75,.99):
            point=(base+alpha*direction).detach().requires_grad_(True)
            p=probabilities(point)[0,target]
            gradient=torch.autograd.grad(p,point)[0]
            derivative=float((gradient*direction).sum())
            eps=.001
            with torch.no_grad():
                finite_difference=float((probabilities(base+(alpha+eps)*direction)[0,target]-probabilities(base+(alpha-eps)*direction)[0,target])/(2*eps))
            difference=abs(derivative-finite_difference)
            output.append(dict(baseline=name,alpha=alpha,probability=float(p.detach()),autograd_derivative=derivative,
                finite_difference=finite_difference,absolute_difference=difference,
                passed=difference<=.01+.05*abs(finite_difference)))
    report={'scope':'Synthetic embedding-path derivative check, not scientific attribution validation',
        'revision':v.revision,'input':text,'dtype':'float32','model_mode':'eval','finite_difference_step':.001,
        'input_ids_vs_inputs_embeds_max_probability_difference':parity,
        'derivative_tolerance':'absolute <= .01 + .05*abs(finite difference); project diagnostic tolerance',
        'all_checks_passed':parity<=1e-6 and all(r['passed'] for r in output),'points':output}
    path=Path(__file__).resolve().parents[1]/'evaluation/xai-gradient-check.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':main()
