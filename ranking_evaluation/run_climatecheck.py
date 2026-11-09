"""Compare actual team ranking models on ClimateCheck human-judged pools."""

import argparse
import csv
import json
from pathlib import Path

from ranking_evaluation.evaluate_climatecheck import evaluate
from ranking_evaluation.pipeline import ModelDescription
from ranking_evaluation.reranker import LocalCrossEncoder
from ranking_evaluation.run_csfcube import run
from shared.datasets.climatecheck import load
from shared.datasets.setup import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/climatecheck'))
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--encoder-path', type=Path, required=True)
    parser.add_argument('--encoder-id', required=True)
    parser.add_argument('--encoder-revision', required=True)
    parser.add_argument('--reranker-path', type=Path, required=True)
    parser.add_argument('--reranker-id', required=True)
    parser.add_argument('--reranker-revision', required=True)
    parser.add_argument('--cpu-float64', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new directory')
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(str(args.encoder_path), device='cpu', local_files_only=True,
                                  trust_remote_code=False)
    if args.cpu_float64:
        encoder.double()
    scorer = LocalCrossEncoder(args.reranker_path, device='cpu', cpu_float64=args.cpu_float64, batch_size=16)
    model = ModelDescription(model_id=args.reranker_id, revision=args.reranker_revision,
        device='cpu', dtype='float64' if args.cpu_float64 else 'default', max_length=512, batch_size=16)
    report = run(load(args.data, split=args.split), encoder=encoder, model_name=args.encoder_id,
                 scorer=scorer, model_description=model, evaluator=evaluate, batch_size=16)
    report['task'] = 'closed human-judged abstract-pool reranking; not official full-corpus retrieval'
    report['provenance'] = {'encoder_id': args.encoder_id, 'encoder_revision': args.encoder_revision,
        'runner_sha256': file_hash(Path(__file__)),
        'shared_runner_sha256': file_hash(Path('ranking_evaluation/run_csfcube.py')),
        'adapter_sha256': file_hash(Path('shared/datasets/climatecheck.py')),
        'evaluator_sha256': file_hash(Path('ranking_evaluation/evaluate_climatecheck.py')),
        'model_files': {name: {p.relative_to(path).as_posix(): file_hash(p)
                for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(path).parts}
            for name, path in [('encoder', args.encoder_path), ('reranker', args.reranker_path)]}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    rows = [{'method': name, **result['binary_metrics']['macro']} for name, result in report['methods'].items()]
    with (args.output_dir / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output_dir / 'summary.csv')


if __name__ == '__main__':
    main()
