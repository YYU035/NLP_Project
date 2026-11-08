"""Explicit local PDF/full-text selection with page references and input budget."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
from ranking_evaluation.frontend_bridge import load_selector,load_verifier_tokenizer
from ranking_evaluation.pdf_evidence import extract_pdf,select_full_text



def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pdf',type=Path,required=True);ap.add_argument('--paper-id',required=True)
    ap.add_argument('--claim-file',type=Path,required=True);ap.add_argument('--source-url')
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--model-path',type=Path,required=True)
    ap.add_argument('--context-radius',type=int,choices=[0,1],default=1)
    ap.add_argument('--verifier-tokenizer',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():ap.error('Output exists')
    claim=a.claim_file.read_text()
    document=extract_pdf(a.pdf,a.paper_id,source_url=a.source_url)
    selector=load_selector(str(a.config.resolve()),str(a.model_path.resolve()))
    selector=copy.copy(selector);selector.config=replace(selector.config,context_radius=a.context_radius)
    tokenizer=load_verifier_tokenizer(str(a.verifier_tokenizer.resolve()))
    result=select_full_text(claim,document,selector,tokenizer)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as h:json.dump(result,h,indent=2,ensure_ascii=False);h.write('\n')
    print(a.output)


if __name__=='__main__':main()
