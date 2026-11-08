"""Re-score frozen predictions after actual Person 4 pair-token preflight."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from ranking_evaluation.evidence_selection import EvidenceSpan
from ranking_evaluation.evaluate_evidence import score_selection,summarize
from ranking_evaluation.frontend_bridge import load_verifier_tokenizer
from ranking_evaluation.scifact import load_scifact,sha256
from ranking_evaluation.verifier_budget import enforce_verifier_budget,VERIFIER_CONTRACT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['corpus','claims','predictions','verifier-tokenizer','output-dir']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    docs,claims=load_scifact(a.corpus,a.claims,split='dev')
    tokenizer=load_verifier_tokenizer(str(a.verifier_tokenizer.resolve()))
    groups=defaultdict(list);checked=0;rejected=0;maximum=0
    with a.predictions.open() as source,(a.output_dir/'per-claim.jsonl').open('x') as out:
        for line in source:
            row=json.loads(line);cid=row['claim_id'];claim=claims[cid]
            spans=[EvidenceSpan(**{**s,'sentence_indices':tuple(s['sentence_indices'])}) for s in row['passages']]
            for span in spans:
                if ' '.join(docs[span.paper_id].abstract)[span.start:span.end]!=span.text:
                    raise ValueError('Predicted passage text differs from original corpus')
            result=enforce_verifier_budget(claim.claim,spans,tokenizer)
            checked+=len(result.checked);rejected+=len(result.rejected)
            maximum=max([maximum]+[c['pair_tokens'] for c in result.checked if c['accepted']])
            predicted={(s.paper_id,i) for s in result.accepted for i in s.sentence_indices}
            row.update(score_selection(predicted,claim.evidence))
            row.update(passages=[asdict(s) for s in result.accepted],predicted_sentence_ids=sorted(predicted),
                       budget_checks=list(result.checked),budget_rejections=list(result.rejected))
            groups[(row['policy'],row['ranking_method'],row['paper_depth'])].append(row)
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
    summaries=[]
    for (policy,method,depth),records in groups.items():
        if {r['claim_id'] for r in records}!=set(claims):raise ValueError('Incomplete evaluation group')
        summaries.append({'policy':policy,'ranking_method':method,'paper_depth':depth,**summarize(records)})
    (a.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (a.output_dir/'manifest.json').write_text(json.dumps({'status':'complete','checked_passage_instances':checked,
        'rejected_instances':rejected,'maximum_accepted_pair_tokens':maximum,'verifier_contract':VERIFIER_CONTRACT,
        'corpus_sha256':sha256(a.corpus),'claims_sha256':sha256(a.claims),'predictions_sha256':sha256(a.predictions),
        'source_sha256':{p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('verifier_budget.py'))},
        'note':'Same frozen choices; complete over-budget passages rejected and metrics recomputed; no dev retuning'},indent=2)+'\n')
    print('Checked',checked,'rejected',rejected,'maximum accepted',maximum)


if __name__=='__main__':main()
