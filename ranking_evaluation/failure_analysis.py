"""Reference-backed failure cases for evidence selection, without invented labels."""
import argparse
import json
from pathlib import Path

from ranking_evaluation.scifact import load_scifact,sha256


def analyze(corpus,claims_file,predictions,output):
    docs,claims=load_scifact(corpus,claims_file,split='dev')
    single,context={},{}
    with predictions.open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['ranking_method']=='rrf' and row['paper_depth']==5:
                (single if row['policy']=='semantic_sentence' else context)[row['claim_id']]=row
    if set(single)!=set(claims) or set(context)!=set(claims):raise ValueError('Prediction coverage mismatch')
    categories={name:[] for name in ['relevant_paper_irrelevant_passages','incomplete_evidence','missing_context_rescued','missing_gold_paper']}
    for cid,claim in claims.items():
        if not claim.evidence:continue
        selected={tuple(pair) for pair in single[cid]['predicted_sentence_ids']}
        expanded={tuple(pair) for pair in context[cid]['predicted_sentence_ids']}
        for doc,rationales in claim.evidence.items():
            gold={i for r in rationales for i in r.sentences}
            chosen={i for d,i in selected if d==doc}
            neighbour={i for d,i in expanded if d==doc}
            complete=any(set(r.sentences)<=chosen for r in rationales)
            example={'claim_id':cid,'claim':claim.claim,'paper_id':doc,'paper_title':docs[doc].title,
                'gold_rationale_alternatives':[r.sentences for r in rationales],
                'selected_sentence_ids':sorted(chosen),'context_sentence_ids':sorted(neighbour),
                'gold_sentences':{str(i):docs[doc].abstract[i] for i in sorted(gold)},
                'selected_sentences':{str(i):docs[doc].abstract[i] for i in sorted(chosen)},
                'added_context_sentences':{str(i):docs[doc].abstract[i] for i in sorted(neighbour-chosen)},
                'source_reference':f'SciFact corpus document {doc}; original sentence indices',
                'assessment':'annotation-based diagnostic; not an independent scientific judgement'}
            if doc not in single[cid]['selected_papers']:
                categories['missing_gold_paper'].append(example)
                continue
            if chosen and not chosen&gold:
                categories['relevant_paper_irrelevant_passages'].append(example)
            if not complete and any(0<len(chosen&set(r.sentences))<len(set(r.sentences)) for r in rationales):
                categories['incomplete_evidence'].append(example)
            if not complete and any(set(r.sentences)<=neighbour for r in rationales):
                categories['missing_context_rescued'].append(example)
    output.mkdir(parents=True,exist_ok=False)
    summary={name:len(rows) for name,rows in categories.items()}
    result={'scope':'RRF top5, frozen semantic sentence vs context; claim-document units',
        'counts':summary,'corpus_sha256':sha256(corpus),'claims_sha256':sha256(claims_file),
        'predictions_sha256':sha256(predictions),'cases':categories,
        'limitations':['Categories can overlap','Context rescue is annotation coverage, not proof of pronoun/coreference resolution',
                      'Incomplete-evidence category requires a multi-sentence rationale; zero cases remain zero',
                      'Selecting unannotated evidence is scored as a mismatch, not a definitive scientific error']}
    (output/'failure-cases.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# Evidence-selection failure analysis','',result['scope'],'',
        '| Category | Claim-document cases |','| --- | --- |']
    lines += [f'| {name} | {count} |' for name,count in summary.items()]
    for name,rows in categories.items():
        lines += ['', '## '+name, '']
        if not rows:lines += ['No cases observed under this fixed definition.'];continue
        for row in rows[:3]:
            lines += [f"### Claim {row['claim_id']}, document {row['paper_id']}",'',row['claim'],'',
                'Gold alternatives: '+json.dumps(row['gold_rationale_alternatives']),
                'Selected sentence indices: '+json.dumps(row['selected_sentence_ids']),
                'Context sentence indices: '+json.dumps(row['context_sentence_ids']),'',
                'Gold text: '+json.dumps(row['gold_sentences'],ensure_ascii=False),'',
                'Selected text: '+json.dumps(row['selected_sentences'],ensure_ascii=False),'',
                'Added context: '+json.dumps(row['added_context_sentences'],ensure_ascii=False)]
    lines += ['', '## Limits','',*['- '+note for note in result['limitations']]]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(summary)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',type=Path,required=True);ap.add_argument('--claims',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();analyze(a.corpus,a.claims,a.predictions,a.output_dir)
