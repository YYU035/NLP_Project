"""Opt-in server-side frontend bridge for the frozen Person 3 selector."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from threading import RLock

from .semantic_evidence import LocalSemanticSelector, SelectionConfig, person4_passages
from .verifier_budget import enforce_verifier_budget, VERIFIER_CONTRACT

_LOCK = RLock()


def _hash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


@lru_cache(maxsize=2)
def load_selector(config_path, model_path):
    """Paths are trusted server settings, never fields in an uploaded payload."""
    config = json.loads(Path(config_path).read_text())
    root = Path(model_path).resolve()
    manifest = json.loads((root/'download_manifest.json').read_text())
    if manifest['model_id'] != config['model_id'] or manifest['revision'] != config['model_revision']:
        raise ValueError('Evidence model differs from the frozen training configuration')
    for name, expected in manifest['files_sha256'].items():
        path = root/name
        if not path.resolve().is_relative_to(root) or _hash(path) != expected:
            raise ValueError('Evidence model checksum mismatch: '+name)
    return LocalSemanticSelector(root, config=SelectionConfig(**config['selection_config']), device='cpu')


@lru_cache(maxsize=2)
def load_verifier_tokenizer(path):
    marker = Path(path)/'pinned_revision.txt'
    if not marker.is_file() or marker.read_text().strip()!=VERIFIER_CONTRACT['tokenizer_revision']:
        raise ValueError('Verifier tokenizer revision is not the checked Person 4 deployment revision')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def select_abstract_with_diagnostics(claim, identity, abstract, *, top_k=3):
    config = os.environ.get('PERSON3_EVIDENCE_CONFIG')
    model = os.environ.get('PERSON3_EVIDENCE_MODEL_PATH')
    tokenizer_path = os.environ.get('PERSON3_VERIFIER_TOKENIZER_PATH')
    if not config or not model or not tokenizer_path:
        raise RuntimeError('Set PERSON3_EVIDENCE_CONFIG, PERSON3_EVIDENCE_MODEL_PATH and PERSON3_VERIFIER_TOKENIZER_PATH explicitly')
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer')
    with _LOCK:
        selector = load_selector(str(Path(config).resolve()), str(Path(model).resolve()))
        spans = selector.select_abstract(claim, identity, abstract)[:top_k]
        tokenizer = load_verifier_tokenizer(str(Path(tokenizer_path).resolve()))
        decision = enforce_verifier_budget(claim, spans, tokenizer)
        return {'passages': person4_passages(decision.accepted),
                'budget_checks': list(decision.checked), 'rejected_passages': list(decision.rejected),
                'verifier_contract': VERIFIER_CONTRACT}


def select_abstract(claim, identity, abstract, *, top_k=3):
    return select_abstract_with_diagnostics(claim, identity, abstract, top_k=top_k)['passages']
