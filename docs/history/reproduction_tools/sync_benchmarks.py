"""Explicitly snapshot the latest reviewed owner benchmark revision, without merging root folders."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REVISION='90aaca109585c12c877367b1a08fe4dfee4a4de3'

def main():
    def read(name):return subprocess.check_output(['git','show',f'{REVISION}:{name}'],cwd=ROOT)
    catalog_name='shared/datasets/evaluation_catalog.json'
    catalog=json.loads(read(catalog_name))
    paths={catalog_name,'shared/APP_EVALUATION_INTEGRATION.md'}
    def walk(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k.endswith('_path') and isinstance(v,str):paths.add(v)
                else:walk(v)
        elif isinstance(obj,list):
            for v in obj:walk(v)
    walk(catalog)
    dest=ROOT/'benchmarks';hashes={}
    for name in sorted(paths):
        target=(dest/name).resolve()
        if not target.is_relative_to(dest.resolve()):raise ValueError('Snapshot path escape')
        raw=read(name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        hashes[name]=hashlib.sha256(raw).hexdigest()
    (dest/'SOURCE.json').write_bytes((json.dumps({'repo':'YYU035/NLP_Project','revision':REVISION,'paths':sorted(paths),
        'scope':'Published owner experiments; frontend integrity validation, not independent full reproduction','sha256':hashes},indent=2)+'\n').encode('utf-8'))
    print(f'Snapshotted{len(paths)} assets at{REVISION}')
if __name__=='__main__':main()
