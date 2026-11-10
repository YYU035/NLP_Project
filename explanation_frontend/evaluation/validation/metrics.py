"""Independent transparent metrics; no labels are supplied to scoring functions."""
import math,random,statistics
import numpy as np

LABELS=('SUPPORTED','CONTRADICTED','INSUFFICIENT')

def ranking_metrics(ranked, relevant, grades=None):
    if len(ranked)!=len(set(ranked)):raise ValueError('Duplicate ranking IDs')
    relevant=set(relevant)
    if not relevant:return None
    result={}
    for k in (5,10,20):
        head=ranked[:k];hits=[i for i,p in enumerate(head,1) if p in relevant]
        result[f'Recall@{k}']=len(hits)/len(relevant)
        result[f'MRR@{k}']=1/min(hits) if hits else 0.
        dcg=sum(1/math.log2(i+1) for i in hits)
        ideal=sum(1/math.log2(i+1) for i in range(1,min(k,len(relevant))+1))
        result[f'nDCG@{k}']=dcg/ideal if ideal else 0.
        if grades is not None:
            gains=[2**grades.get(p,0)-1 for p in head]
            ideal_gains=sorted([2**g-1 for g in grades.values()],reverse=True)[:k]
            ideal_dcg=sum(g/math.log2(i+2) for i,g in enumerate(ideal_gains))
            result[f'graded_nDCG@{k}']=sum(g/math.log2(i+2) for i,g in enumerate(gains))/ideal_dcg if ideal_dcg else 0.
    return result

def classification(rows):
    if not rows:return {'n':0,'accuracy':None,'macro_f1':None,'per_class':{}}
    if any(r['gold'] not in LABELS or r.get('prediction') not in (*LABELS,'CONFLICTING',None) for r in rows):raise ValueError('Unknown classification label')
    predicted=(*LABELS,'CONFLICTING','PROCESSING_FAILURE')
    matrix={g:{p:sum(r['gold']==g and (r.get('prediction') or 'PROCESSING_FAILURE')==p for r in rows) for p in predicted} for g in LABELS}
    classes={}
    for label in LABELS:
        tp=matrix[label][label];actual=sum(matrix[label].values());chosen=sum(matrix[g][label] for g in LABELS)
        precision=tp/chosen if chosen else 0.;recall=tp/actual if actual else 0.
        classes[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.,'support':actual}
    return {'n':len(rows),'accuracy':sum(r['gold']==r.get('prediction') for r in rows)/len(rows),
        'macro_f1':statistics.mean(v['f1'] for v in classes.values() if v['support']>0),'macro_classes':[k for k,v in classes.items() if v['support']>0],'per_class':classes,'confusion_matrix':matrix,
        'macro_scope':'Macro over gold-present declared classes (all3 in the complete main cohort,2 in positive-only controls); CONFLICTING/operational failures count as errors.'}

def evidence_score(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return {'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}

def bootstrap(values,clusters=None,repeats=2000,seed=20261110):
    if not values:return {'n':0,'estimate':None,'low':None,'high':None}
    values=np.asarray(values,dtype=float)
    if not np.all(np.isfinite(values)):raise ValueError('Nonfinite metric')
    if clusters is None:clusters=list(range(len(values)))
    if len(clusters)!=len(values) or repeats<=0:raise ValueError('Invalid bootstrap inputs')
    groups={}
    for i,key in enumerate(clusters):groups.setdefault(str(key),[]).append(i)
    arrays=[values[idx] for idx in groups.values()];rng=np.random.default_rng(seed);samples=[]
    for _ in range(repeats):
        chosen=rng.integers(0,len(arrays),len(arrays));samples.append(float(np.concatenate([arrays[j] for j in chosen]).mean()))
    return {'n':len(values),'clusters':len(arrays),'estimate':float(values.mean()),'low':float(np.quantile(samples,.025)),
            'high':float(np.quantile(samples,.975)),'repeats':repeats,'seed':seed,'method':'percentile cluster bootstrap; dataset dependence limitations still apply'}

def reliability(rows):
    if any(not math.isfinite(r['confidence']) or not 0<=r['confidence']<=1 for r in rows):raise ValueError('Invalid confidence')
    bins=[];ece=0.
    for index in range(10):
        lo=index/10
        subset=[r for r in rows if min(int(r['confidence']*10),9)==index]
        if not subset:continue
        accuracy=statistics.mean(r['correct'] for r in subset);confidence=statistics.mean(r['confidence'] for r in subset)
        ece+=len(subset)*abs(accuracy-confidence)/len(rows)
        bins.append({'lower':lo,'n':len(subset),'accuracy':accuracy,'mean_confidence':confidence})
    selective=[]
    for cutoff in (0.,.5,.7,.9,.95):
        subset=[r for r in rows if r['confidence']>=cutoff]
        selective.append({'threshold':cutoff,'coverage':len(subset)/len(rows) if rows else 0.,'accuracy':statistics.mean(r['correct'] for r in subset) if subset else None})
    return {'n':len(rows),'ece_10_bins':ece if rows else None,'bins':bins,'selective_accuracy':selective,
            'scope':'Raw passage argmax against annotated SUPPORT/CONTRADICT rationale targets only; no neutral-gold cohort and no four-class aggregate calibration claim.'}
