"""Bounded, exact word-MASK score differences; not Integrated Gradients."""
import math,time


def explain_occlusion(verifier,claim,passage,*,target=None,max_words=96,batch_size=4):
    import torch
    if not 1<=max_words<=512 or not 1<=batch_size<=8:raise ValueError('Invalid occlusion budget')
    text=getattr(passage,'text',passage)
    if not text.strip() or not claim.strip():raise ValueError('Empty claim or evidence')
    if verifier.model.training:raise ValueError('Explanation requires model.eval()')
    tokenizer=verifier.tokenizer
    if not tokenizer.is_fast:raise ValueError('Exact word offsets require a fast tokenizer')
    if tokenizer.mask_token_id is None:raise ValueError('Tokenizer has no MASK token')
    if len(tokenizer.encode(text,claim,add_special_tokens=True,truncation=False))>verifier.max_length:
        raise ValueError('Occlusion input exceeds the verifier budget')
    enc=tokenizer(text,claim,return_tensors='pt',return_offsets_mapping=True)
    offsets=enc.pop('offset_mapping')[0].tolist();sequence=enc.sequence_ids(0);word_ids=enc.word_ids(0)
    groups=[]
    for index,(segment,word_id) in enumerate(zip(sequence,word_ids)):
        if segment is None:continue
        key=(segment,word_id)
        if not groups or groups[-1]['key']!=key:groups.append({'key':key,'segment':segment,'indices':[],'start':offsets[index][0],'end':offsets[index][1]})
        groups[-1]['indices'].append(index);groups[-1]['start']=min(groups[-1]['start'],offsets[index][0]);groups[-1]['end']=max(groups[-1]['end'],offsets[index][1])
    # Balance evidence/claim coverage before spending the remainder; no importance-based selection.
    by_segment=[[g for g in groups if g['segment']==segment] for segment in (0,1)]
    chosen=[]
    while len(chosen)<min(len(groups),max_words):
        for group in by_segment:
            if group and len(chosen)<max_words:chosen.append(group.pop(0))
    tensors={k:v.to(verifier.device) for k,v in enc.items()}
    def probs(inputs):return verifier.model(**inputs).logits.float().softmax(-1)
    started=time.perf_counter()
    with torch.no_grad():
        original=probs(tensors)[0];predicted=int(original.argmax())
        idx=predicted if target is None else next(i for i,label in verifier.label_mapping.items() if label.value==str(getattr(target,'value',target)))
        probability=float(original[idx]);scores=[]
        for begin in range(0,len(chosen),batch_size):
            chunk=chosen[begin:begin+batch_size]
            batch={k:v.expand(len(chunk),-1).clone() for k,v in tensors.items()}
            for row,group in enumerate(chunk):batch['input_ids'][row,group['indices']]=tokenizer.mask_token_id
            scores.extend(probability-float(p) for p in probs(batch)[:,idx])
    scale=max(.01,max([abs(x) for x in scores],default=0.))
    words=[]
    for group,score in zip(chosen,scores):
        raw=(text if group['segment']==0 else claim)[group['start']:group['end']]
        start=group['start']+len(raw)-len(raw.lstrip())
        words.append({'text':raw.strip(),'segment':'evidence' if group['segment']==0 else 'claim','start':start,'end':group['end'],
                      'attribution':score,'normalized':score/scale,'token_indices':group['indices']})
    if not all(math.isfinite(w['attribution']) for w in words):raise ValueError('Nonfinite occlusion score')
    return {'kind':'word_occlusion','claim':claim,'evidence_text':text,'paper_id':getattr(passage,'paper_id',''),
        'passage_id':getattr(passage,'passage_id',''),'target_label':verifier.label_mapping[idx].value,
        'predicted_label':verifier.label_mapping[predicted].value,'target_class':verifier.model.config.id2label[idx],
        'target_probability':probability,'words':words,'score_status':'measured','coverage':{'scored_words':len(words),'total_words':len(groups),'complete':len(words)==len(groups)},
        'settings':{'method':'Word-level MASK occlusion','baseline':'One whole word replaced by MASK; special tokens and original attention mask fixed',
            'checkpoint':verifier.checkpoint,'revision':getattr(verifier,'revision',None),'max_words':max_words,'batch_size':batch_size,
            'target_selection':'gold class' if target is not None else 'raw argmax before confidence threshold'},
        'seconds':time.perf_counter()-started,'limitations':['Conditional model score differences, not scientific truth or causal proof.',
            'MASK perturbations can be unnatural; interacting words are not additive. No IG completeness guarantee applies.']}
