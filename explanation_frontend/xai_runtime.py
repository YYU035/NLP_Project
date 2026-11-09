"""Frontend IG reliability policy; owner attribution implementation stays pinned."""
import math
import time

from ._upstream.xai.explain_ig import IGExplainer

RELATIVE_LIMIT = .05
ABSOLUTE_LIMIT = .001


def reliability(result):
    """Require absolute AND relative completeness; a tiny denominator never passes alone."""
    gap = abs(result['target_probability'] - result['baseline_probability'])
    error = abs(result['convergence_delta'])
    values = [gap, error, result['baseline_probability'], result['target_probability']]
    values += [w[k] for w in result['words'] for k in ('attribution', 'normalized')]
    finite = all(math.isfinite(v) for v in values)
    relative = error / max(gap, 1e-8) if finite else None
    passed = finite and not result['truncated'] and error <= ABSOLUTE_LIMIT and relative <= RELATIVE_LIMIT
    return dict(relative_completeness_error=relative, absolute_completeness_error=error,
                numerical_status='passed' if passed else 'unstable', finite=finite,
                gate={'max_relative_error':RELATIVE_LIMIT,'max_absolute_error':ABSOLUTE_LIMIT})


def explain_checked(verifier, claim, passage, *, baseline='mask', steps=(32,128,512),
                    target=None, sentences=None, return_run=False):
    """Fixed baseline; retry quadrature without changing target or replacing failed scores."""
    if baseline not in {'pad','mask','unk'}:
        raise ValueError('Unsupported baseline')
    if not steps or any(n < 2 or n > 2048 for n in steps):
        raise ValueError('Steps must be between 2 and 2048')
    text = ' '.join(sentences) if sentences is not None else getattr(passage,'text',passage)
    if len(verifier.tokenizer.encode(text,claim,add_special_tokens=True,truncation=False)) > verifier.max_length:
        raise ValueError('Attribution input exceeds the original verifier budget')
    attempts=[]
    for n in steps:
        started=time.perf_counter()
        explainer=IGExplainer(verifier,n_steps=n,internal_batch_size=1,baseline=baseline)
        run=explainer._run(text,claim,target)
        from ._upstream.claim_verifier.schemas import EvidencePassage
        native=passage if not isinstance(passage,str) else EvidencePassage(
            passage_id='diagnostic',paper_id='diagnostic',section='abstract',text=passage)
        exp=explainer._build_explanation(run,native,claim,text,sentences)
        result=exp.model_dump(mode='json')
        result.update(reliability(result))
        attempts.append({'steps':n,'seconds':time.perf_counter()-started,
                         'absolute_error':result['absolute_completeness_error'],
                         'relative_error':result['relative_completeness_error'],
                         'numerical_status':result['numerical_status']})
        if not result['finite']:
            raise ValueError('Attribution produced nonfinite values')
        if result['numerical_status']=='passed':
            break
    result['attempts']=attempts
    result['settings']['baseline_name']=baseline
    result['settings']['model_revision']=getattr(verifier,'revision',None)
    result['settings']['attention_mask']='original input mask held fixed along embedding path'
    result['settings']['target_selection']='gold label' if target is not None else 'raw argmax before confidence threshold'
    result['interpretation']='Signed contribution to target probability relative to the chosen baseline; not scientific truth or a causal proof.'
    return (result,explainer,run) if return_run else result
