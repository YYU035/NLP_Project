"""Run four team retrieval/ranking methods on all RECIPER source-paper tasks."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

from ranking_evaluation.evaluate_reciper import evaluate
from ranking_evaluation.pipeline import ModelDescription
from ranking_evaluation.reranker import LocalCrossEncoder
from ranking_evaluation.run_csfcube import run
from shared.datasets.reciper import load, prepare
from shared.datasets.setup import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/reciper'))
    parser.add_argument('--encoder-path', type=Path, required=True)
    parser.add_argument('--encoder-id', required=True)
    parser.add_argument('--encoder-revision', required=True)
    parser.add_argument('--reranker-path', type=Path, required=True)
    parser.add_argument('--reranker-id', required=True)
    parser.add_argument('--reranker-revision', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--cpu-float64', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--rerank-top-n', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new directory')
    if args.threads <= 0 or args.batch_size <= 0 or args.rerank_top_n <= 0:
        parser.error('Threads, batch size and rerank depth must be positive')
    if args.cpu_float64 and args.device != 'cpu':
        parser.error('CPU float64 requires CPU execution')
    dataset = load(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prepare(dataset, args.output_dir / 'inputs')
    status_path = args.output_dir / 'status.json'
    def status(value):
        status_path.write_text(json.dumps(value, indent=2) + '\n')
    status({'status': 'loading_models', 'dataset_version': dataset['version']})
    started = perf_counter()
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(args.threads)
        encoder = SentenceTransformer(str(args.encoder_path), device=args.device,
                                      local_files_only=True, trust_remote_code=False)
        if args.cpu_float64:
            encoder.double()
        scorer = LocalCrossEncoder(args.reranker_path, device=args.device,
            cpu_float64=args.cpu_float64, batch_size=args.batch_size, max_length=512)
        model = ModelDescription(model_id=args.reranker_id, revision=args.reranker_revision,
            device=args.device, dtype=str(next(scorer.model.model.parameters()).dtype).removeprefix('torch.'),
            max_length=512, batch_size=args.batch_size)
        model_load_seconds = perf_counter() - started
        status({'status': 'running', 'completed_queries': 0, 'total_queries': len(dataset['queries'])})
        def progress(done, total, identity):
            if done % 32 == 0 or done == total:
                update = {'status': 'running', 'completed_queries': done, 'total_queries': total,
                          'last_query_id': identity, 'seconds_since_model_loading_started': perf_counter() - started}
                status(update)
                print(json.dumps(update), flush=True)
        report = run(dataset, encoder=encoder, model_name=args.encoder_id, scorer=scorer,
            model_description=model, device=args.device, batch_size=args.batch_size,
            top_n=args.rerank_top_n, rrf_k=60, evaluator=evaluate, progress=progress)
        report['task'] = 'full-collection title/abstract source-paper retrieval; no gold candidate preselection'
        report['model_load_seconds'] = model_load_seconds
        report['elapsed_before_serialization_seconds'] = perf_counter() - started
        report['environment'] = {'python': platform.python_version(), 'platform': platform.platform(),
            'torch_threads': torch.get_num_threads(), 'packages': {name: importlib.metadata.version(name)
                for name in ['torch', 'sentence-transformers', 'transformers', 'numpy', 'pydantic']}}
        report['provenance'] = {'encoder_id': args.encoder_id, 'encoder_revision': args.encoder_revision,
            'dataset_manifest': dataset['manifest'],
            'code_hashes': {name: file_hash(Path(name)) for name in [
                'ranking_evaluation/run_reciper.py', 'ranking_evaluation/run_csfcube.py',
                'ranking_evaluation/evaluate_reciper.py', 'shared/datasets/reciper.py',
                'retrieval/bm25.py', 'retrieval/sbert.py', 'ranking_evaluation/pipeline.py',
                'ranking_evaluation/rrf.py', 'ranking_evaluation/reranker.py', 'ranking_evaluation/metrics.py']},
            'model_files': {name: {p.relative_to(path).as_posix(): file_hash(p)
                for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(path).parts}
                for name, path in [('encoder', args.encoder_path), ('reranker', args.reranker_path)]}}
        (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        rows = [{'method': name, 'queries': result['query_count'],
                 **result['binary_metrics']['macro']} for name, result in report['methods'].items()]
        with (args.output_dir / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        status({'status': 'complete', 'completed_queries': len(dataset['queries']),
                'total_queries': len(dataset['queries']), 'dataset_version': dataset['version']})
        print(args.output_dir / 'summary.csv', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        previous = json.loads(status_path.read_text())
        previous.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error_type=type(exc).__name__)
        status(previous)
        raise


if __name__ == '__main__':
    main()
