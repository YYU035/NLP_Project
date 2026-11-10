"""Explicit setup of the exact train-frozen Person 3 embedding revision."""
import hashlib
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / '_upstream/person3/semantic-train-frozen.json'
MODEL_PATH = ROOT / '.models/person3-semantic'


def setup():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))
    model, revision = config['model_id'], config['model_revision']
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    response = session.get(f'https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true', timeout=30)
    response.raise_for_status()
    info = response.json()
    if info['sha'] != revision:
        raise ValueError('Model revision mismatch')
    allowed = {'README.md', 'config.json', 'modules.json', 'sentence_bert_config.json',
        'config_sentence_transformers.json', 'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'vocab.txt', '1_Pooling/config.json', 'model.safetensors'}
    hashes = {}
    for entry in info['siblings']:
        name = entry['rfilename']
        if name not in allowed:
            continue
        path = MODEL_PATH / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.part')
            with session.get(f'https://huggingface.co/{model}/resolve/{revision}/{name}?download=true', timeout=60, stream=True) as response, temporary.open('wb') as handle:
                response.raise_for_status()
                for chunk in response.iter_content(4*1024*1024):
                    if chunk:
                        handle.write(chunk)
            temporary.replace(path)
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        expected = entry.get('lfs', {}).get('sha256')
        if expected and digest != expected:
            raise ValueError('Model checksum mismatch')
        hashes[name] = digest
    if 'model.safetensors' not in hashes:
        raise ValueError('Missing safe weights')
    (MODEL_PATH/'download_manifest.json').write_text(json.dumps({
        'model_id': model, 'revision': revision, 'files_sha256': hashes}, indent=2), encoding='utf-8')
    print('Person 3 semantic model ready:', MODEL_PATH, flush=True)


if __name__ == '__main__':
    setup()
