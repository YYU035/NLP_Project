"""Explicit saved actual-model word-occlusion results; never presented as fresh inference."""
import hashlib,json,math
from pathlib import Path
from .models import digest
ROOT=Path(__file__).parent/'showcase/attributions'


def cache_key(snapshot,passage_id):
    passage=next(p for p in snapshot.verification.evidence_passages if p.passage_id==passage_id)
    model=snapshot.verification.model_settings
    return digest({'claim':snapshot.claim,'passage_id':passage_id,'paper_id':passage.paper_id,'text':passage.text,
                   'checkpoint':model.checkpoint,'revision':model.revision,'method':'word-mask-occlusion-v1'})


def load_recorded(snapshot,passage_id):
    manifest=ROOT/'manifest.json'
    if not manifest.is_file():return None
    item=json.loads(manifest.read_text(encoding='utf-8')).get(cache_key(snapshot,passage_id))
    if item is None:return None
    path=(ROOT/item['file']).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Attribution cache path escape')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Attribution cache checksum mismatch')
    result=json.loads(raw)
    passage=next(p for p in snapshot.verification.evidence_passages if p.passage_id==passage_id)
    if result['kind']!='word_occlusion' or result['claim']!=snapshot.claim or result['evidence_text']!=passage.text:
        raise ValueError('Saved attribution input mismatch')
    if not all(math.isfinite(w['attribution']) and math.isfinite(w['normalized']) for w in result['words']):
        raise ValueError('Saved attribution is nonfinite')
    result['recorded']=True
    result['recording_note']='Saved actual local-model computation; no new inference for this display.'
    return result
