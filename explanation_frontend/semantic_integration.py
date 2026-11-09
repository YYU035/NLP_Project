"""Pinned Person 3 selection and exact Person 4 token-budget handoff."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from ._upstream.person3.semantic_evidence import LocalSemanticSelector, SelectionConfig, person4_passages
from ._upstream.person3.verifier_budget import enforce_verifier_budget, VERIFIER_CONTRACT
from .semantic_setup import CONFIG, MODEL_PATH


@lru_cache(maxsize=1)
def local_selector():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    manifest = json.loads((MODEL_PATH/'download_manifest.json').read_text())
    if manifest['revision'] != config['model_revision'] or manifest['model_id'] != config['model_id']:
        raise ValueError('Semantic model differs from frozen configuration')
    for name, digest in manifest['files_sha256'].items():
        file = (MODEL_PATH/name).resolve()
        if not file.is_relative_to(MODEL_PATH.resolve()):
            raise ValueError('Invalid model manifest path')
        with file.open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest() != digest:
                raise ValueError('Semantic model checksum mismatch')
    return LocalSemanticSelector(MODEL_PATH, config=SelectionConfig(**config['selection_config']))


def select_bounded(claim, identity, abstract, tokenizer, *, selector=None):
    spans = (selector or local_selector()).select_abstract(claim, identity, abstract)
    decision = enforce_verifier_budget(claim, spans, tokenizer)
    return {'passages': person4_passages(decision.accepted),
            'source_spans': [asdict(s) for s in decision.accepted],
            'budget_checks': list(decision.checked), 'rejected_passages': list(decision.rejected),
            'verifier_contract': VERIFIER_CONTRACT,
            'status': 'selected' if decision.accepted else 'budget_rejected' if decision.rejected
                      else 'below_threshold' if abstract and abstract.strip() else 'text_unavailable'}
