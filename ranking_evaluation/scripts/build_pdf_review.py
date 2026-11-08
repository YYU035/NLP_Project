"""Separate PDF review packet and real Person 4 token-limit smoke checks."""

import argparse
import csv
import copy
from dataclasses import replace
import hashlib
import html
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import subprocess

from ranking_evaluation.frontend_bridge import load_selector,load_verifier_tokenizer
from ranking_evaluation.pdf_evidence import extract_pdf,select_full_text
from ranking_evaluation.evidence_selection import EvidenceSpan
from ranking_evaluation.verifier_budget import enforce_verifier_budget,VERIFIER_CONTRACT

CASES=[
 ('t-architecture','transformer','The Transformer replaces recurrent and convolutional layers with attention mechanisms.'),
 ('t-base-schedule','transformer','The Transformer base model was trained for 100,000 steps, about twelve hours, on eight NVIDIA P100 GPUs.'),
 ('t-big-results','transformer','The big Transformer achieved 28.4 BLEU on WMT 2014 English-to-German after 3.5 days of training on eight P100 GPUs.'),
 ('b-bidirectional','bert','BERT conditions on both left and right context in all layers during pretraining.'),
 ('b-masking-qualification','bert','BERT replaces 80 percent of the tokens selected for prediction with the mask token, rather than masking 80 percent of every input.'),
 ('b-hardware','bert','BERT Large was pretrained for four days on sixteen Cloud TPUs containing sixty-four TPU chips.'),
]


def run(args):
    args.output_dir.mkdir(parents=True,exist_ok=False)
    selector=load_selector(str(args.config.resolve()),str(args.model_path.resolve()))
    selector=copy.copy(selector)
    selector.config=replace(selector.config,context_radius=1)
    tokenizer=load_verifier_tokenizer(str(args.verifier_tokenizer.resolve()))
    sources=json.loads((args.pdf_root/'sources.json').read_text())
    documents={}
    for key in ['transformer','bert']:
        source=next(s for s in sources if Path(s['file']).stem==key)
        documents[key]=extract_pdf(source['file'],source['paper_id'],source_url=source['url'])
    sys.path.insert(0,str(args.person4_root/'claim_verifier'))
    import torch
    from transformers import AutoModelForSequenceClassification
    spec=importlib.util.spec_from_file_location('person4_original_verifier',args.person4_root/'claim_verifier/verifier.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    schemas=importlib.import_module('schemas')
    verifier=object.__new__(owner.ClaimVerifier)
    verifier.checkpoint='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
    verifier.revision=(args.verifier_tokenizer/'pinned_revision.txt').read_text().strip()
    verifier.max_length,verifier.threshold,verifier.seed,verifier.device=512,.5,42,'cpu'
    verifier.tokenizer=tokenizer
    torch.set_num_threads(4);torch.manual_seed(42)
    verifier.model=AutoModelForSequenceClassification.from_pretrained(str(args.verifier_tokenizer),
        local_files_only=True,trust_remote_code=False,dtype=torch.float32).eval()
    verifier.label_mapping=verifier._build_label_mapping()
    review_rows=[];case_records=[];page_images={}
    assets=args.output_dir/'pages';assets.mkdir()
    pdfs=args.output_dir/'pdfs';pdfs.mkdir()
    import pymupdf
    for key in documents:shutil.copyfile(args.pdf_root/(key+'.pdf'),pdfs/(key+'.pdf'))
    for case_id,key,claim in CASES:
        document=documents[key]
        selected=select_full_text(claim,document,selector,tokenizer,top_k=2)
        native=[schemas.EvidencePassage.model_validate(p) for p in selected['passages']]
        response=verifier.verify(claim,[{'paper_id':document['paper_id'],'title':key,'evidence_type':'full_text'}],
            {document['paper_id']:native})
        if any(p.truncated for p in response.per_passage_labels):raise ValueError('Accepted passage was truncated by Person 4')
        aggregate=verifier.aggregate(response)
        case_records.append({'case_id':case_id,'source':key,'claim':claim,'selection':selected,
            'verifier_output':response.model_dump(mode='json'),'verdict':aggregate.model_dump(mode='json'),
            'reference_labels':'none; model predictions are not human gold labels'})
        for index,span in enumerate(selected['source_spans']):
            image_key=(key,span['page'])
            if image_key not in page_images:
                name=f'{key}-page-{span["page"]}.png'
                with pymupdf.open(args.pdf_root/(key+'.pdf')) as pdf:
                    pdf[span['page']-1].get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(assets/name)
                page_images[image_key]=name
            check=next(c for c in selected['budget_checks'] if c['passage_id']==span['passage_id'])
            review_rows.append({'sample_id':case_id+f'-{index+1}','case_id':case_id,'claim':claim,
                'paper_id':document['paper_id'],'page':span['page'],'start':span['start'],'end':span['end'],
                'passage_id':span['passage_id'],'pair_tokens':check['pair_tokens'],'passage':span['text'],
                'image':'pages/'+page_images[image_key],'pdf':'pdfs/'+key+'.pdf',
                'reviewer':'','source_matches':'','relevant_to_claim':'','context_sufficient':'',
                'within_input_limit':'','notes':'','reviewed_at':''})
        print(case_id,len(native),'passages, all untruncated',flush=True)
    synthetic=EvidenceSpan('stress',(0,),0,1,'word '*900,1,'stress:long')
    stress=enforce_verifier_budget('A scientific claim.',[synthetic],tokenizer)
    assert not stress.accepted and stress.rejected
    manifest_path=args.person4_root/'source_manifest.json'
    if manifest_path.exists():
        person4_source=json.loads(manifest_path.read_text())
    else:
        head=subprocess.run(['git','rev-parse','HEAD'],cwd=args.person4_root,capture_output=True,text=True)
        person4_source={'commit':head.stdout.strip() if head.returncode==0 else 'unknown',
            'files_sha256':{name:hashlib.sha256((args.person4_root/name).read_bytes()).hexdigest() for name in ['claim_verifier/verifier.py','claim_verifier/schemas.py']}}
    summary={'status':'automated_checks_complete','human_review_status':'pending',
        'pdf_documents':len(documents),'claim_cases':len(case_records),'passages':len(review_rows),
        'all_accepted_pairs_within_512':all(r['pair_tokens']<=512 for r in review_rows),
        'real_person4_truncated_inputs':0,'stress_case_rejected':list(stress.rejected),
        'person4_source':person4_source,
        'verifier_contract':VERIFIER_CONTRACT,'sources':sources,
        'limitations':['Six constructed review claims, not an independent accuracy benchmark',
            'Human review CSV is deliberately blank; no human approval is invented',
            'PDF column/reading-order/figure issues require the separate visual review',
            'The verifier initializer uses pinned local weights; original Person 4 classify/verify methods are unchanged']}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output_dir/'cases.json').write_text(json.dumps(case_records,indent=2,ensure_ascii=False)+'\n')
    with (args.output_dir/'human-review.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(review_rows[0]));writer.writeheader();writer.writerows(review_rows)
    chunks=['<!doctype html><meta charset="utf-8"><title>Person 3 PDF review</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:16px}article{border-top:2px solid #bbb;margin-top:40px;padding-top:20px}pre{white-space:pre-wrap;background:#f3f5f8;padding:18px}img{width:100%;border:1px solid #ddd}th,td{padding:8px;text-align:left}strong{color:#923}</style>',
        '<h1>Separate PDF evidence review</h1><p><strong>Human review pending.</strong> Automated exact-text/token checks do not establish relevance or scientific correctness.</p>',
        '<p>Open the linked PDF page, compare the extracted passage with the image, then fill reviewer, source_matches, relevant_to_claim, context_sufficient, within_input_limit and notes in <a href="human-review.csv">human-review.csv</a>. Use yes/no/uncertain. Do not infer human labels from the model verdict.</p>',
        '<p>These are six constructed claims against two fixed-version research PDFs. They are a review sample, not a second annotated benchmark dataset.</p>']
    for row in review_rows:
        chunks += ['<article><h2>'+html.escape(row['sample_id'])+'</h2><p>'+html.escape(row['claim'])+'</p>',
            '<p>Paper '+html.escape(row['paper_id'])+'; PDF page '+str(row['page'])+'; source offsets '+str(row['start'])+'–'+str(row['end'])+'; verifier pair tokens '+str(row['pair_tokens'])+'/512.</p>',
            '<p><a href="'+html.escape(row['pdf'])+'#page='+str(row['page'])+'">Open original PDF page</a></p>',
            '<pre>'+html.escape(row['passage'])+'</pre>',
            '<img loading="lazy" src="'+html.escape(row['image'])+'" alt="Original PDF page"><p>Reviewer decision: blank — record it in the CSV.</p></article>']
    (args.output_dir/'index.html').write_text('\n'.join(chunks))
    from .enhance_pdf_review import render
    render(args.output_dir)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['pdf-root','config','model-path','verifier-tokenizer','person4-root','output-dir']:
        ap.add_argument('--'+name,type=Path,required=True)
    run(ap.parse_args())
