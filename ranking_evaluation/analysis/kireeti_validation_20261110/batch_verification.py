"""Batch only the model forward pass; retain the owner's exact label conversion."""
from types import SimpleNamespace
import torch


def classify_batched(verifier,pairs,*,batch_size=4):
    """Return owner PassageVerdicts in input order without changing pair text."""
    if type(batch_size) is not int or batch_size<1:raise ValueError('Positive batch size required')
    model=verifier.model
    owner=type(verifier)._classify_passage
    result=[None]*len(pairs)
    ordered=sorted(range(len(pairs)),key=lambda i:len(pairs[i][1].text))
    try:
        for first in range(0,len(ordered),batch_size):
            indices=ordered[first:first+batch_size]
            claims=[pairs[i][0] for i in indices];passages=[pairs[i][1].text for i in indices]
            inputs=verifier.tokenizer(passages,claims,padding=True,truncation=False,return_tensors='pt').to(verifier.device)
            if inputs['input_ids'].shape[1]>verifier.max_length:raise ValueError('Batch contains over-budget pair')
            with torch.no_grad():logits=model(**inputs).logits
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite batched logits')
            for j,i in enumerate(indices):
                # Owner code tokenizes and applies its original softmax, mapping,
                # uncertainty threshold and operational-status rules unchanged.
                class FixedLogits:
                    config=model.config
                    def __call__(self,**kwargs):return SimpleNamespace(logits=logits[j:j+1])
                verifier.model=FixedLogits()
                result[i]=owner(verifier,pairs[i][0],pairs[i][1])
            verifier.model=model
    finally:verifier.model=model
    return result
