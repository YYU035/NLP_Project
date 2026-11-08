"""Produce source-linked Person 4 passages from a complete RankingOutput."""

import argparse
import json
from pathlib import Path

from ranking_evaluation.frontend_bridge import load_selector
from ranking_evaluation.pipeline import RankingOutput
from ranking_evaluation.semantic_evidence import select_ranked_evidence


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--model-path',type=Path,required=True)
    ap.add_argument('--verifier-tokenizer',type=Path,required=True)
    ap.add_argument('--paper-depth',type=int,default=5)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Output exists; choose a new path')
    if args.paper_depth<=0:ap.error('paper-depth must be positive')
    ranking=RankingOutput.model_validate_json(args.input.read_bytes())
    selector=load_selector(str(args.config.resolve()),str(args.model_path.resolve()))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(args.verifier_tokenizer.resolve()),local_files_only=True,trust_remote_code=False)
    result=select_ranked_evidence(ranking,selector,tokenizer=tokenizer,paper_depth=args.paper_depth)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n')
    print(args.output)


if __name__=='__main__':main()
