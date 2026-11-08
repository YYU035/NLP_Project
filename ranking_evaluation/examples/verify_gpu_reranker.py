"""CUDA smoke check on constructed texts; not a retrieval benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from time import perf_counter

from ranking_evaluation.reranker import LocalCrossEncoder
from ranking_evaluation.scripts.setup_reranker import MODEL_ID, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; GPU smoke check requires CUDA')
    manifest = json.loads((args.model_path / 'download_manifest.json').read_text())
    if manifest['model_id'] != MODEL_ID or manifest['revision'] != REVISION:
        raise RuntimeError('Model manifest does not match pinned revision')
    for name, expected in manifest['files'].items():
        with (args.model_path / name).open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                raise RuntimeError(f'Model checksum mismatch: {name}')
    pairs = [
        ('How many people live in Berlin?', 'Bananas are a tropical fruit used in cooking.'),
        ('How many people live in Berlin?', 'Berlin has a population of about 3.5 million people.'),
        ('How many people live in Berlin?', 'Berlin has many art museums and historical buildings.'),
    ]

    def timed(operation):
        torch.cuda.synchronize()
        start = perf_counter()
        result = operation()
        torch.cuda.synchronize()
        return result, perf_counter() - start

    scorer, load_seconds = timed(lambda: LocalCrossEncoder(
        args.model_path, device='cuda', batch_size=3,
    ))
    if next(scorer.model.model.parameters()).device.type != 'cuda':
        raise RuntimeError('Model parameters are not on CUDA')
    cold, cold_seconds = timed(lambda: scorer.score_pairs(pairs))
    warm, warm_seconds = timed(lambda: scorer.score_pairs(pairs))
    for scores in (cold, warm):
        values = [float(value) for value in scores]
        if len(values) != len(pairs) or not all(math.isfinite(value) for value in values):
            raise RuntimeError('Expected one finite score per pair')
        if max(range(len(values)), key=values.__getitem__) != 1:
            raise RuntimeError('Expected population passage to rank first')
    report = {
        'type': 'CUDA smoke check on constructed texts; not benchmark',
        'model_id': MODEL_ID, 'revision': REVISION, 'weight_manifest': manifest,
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in (
            'torch', 'sentence-transformers', 'transformers', 'huggingface-hub')},
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'dtype': str(next(scorer.model.model.parameters()).dtype),
        'batch_size': 3, 'max_length': 512,
        'load_seconds': load_seconds, 'cold_seconds': cold_seconds,
        'warm_seconds': warm_seconds, 'scores': [float(value) for value in warm],
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
