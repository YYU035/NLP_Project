"""Pinned, bounded dataset staging inside the owned folder only."""
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    from evaluation.validation.dataset_adapters.setup import SOURCES, verify_download
    import requests
    target_root=(ROOT/'outputs/validation/data').resolve()
    if not target_root.is_relative_to(ROOT.resolve()):raise ValueError('Invalid data root')
    for name in ('csfcube','reciper'):
        destination=target_root/name;source=SOURCES[name]
        if (destination/'download_manifest.json').is_file():
            verify_download(destination,name);print('Verified cached',name,flush=True);continue
        destination.mkdir(parents=True,exist_ok=True);hashes={}
        for file in source['files']:
            url=(f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{file}" if source.get('host')=='huggingface' else f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{file}")
            path=(destination/file).resolve()
            if not path.is_relative_to(destination.resolve()):raise ValueError('Dataset path escape')
            path.parent.mkdir(parents=True,exist_ok=True)
            for attempt in range(3):
                try:
                    with requests.get(url,timeout=60,stream=True) as response:
                        response.raise_for_status();total=0
                        with path.open('wb') as handle:
                            for chunk in response.iter_content(1024*1024):
                                total+=len(chunk)
                                if total>50*1024*1024:raise ValueError('File exceeds50MiB')
                                handle.write(chunk)
                    break
                except requests.RequestException:
                    if attempt==2:raise
                    time.sleep(1+attempt)
            hashes[file]=hashlib.sha256(path.read_bytes()).hexdigest();print(name,file,path.stat().st_size,flush=True)
        (destination/'download_manifest.json').write_text(json.dumps({'dataset':name,'repository':source['repository'],'revision':source['revision'],'files_sha256':hashes},indent=2)+'\n',encoding='utf-8')
        verify_download(destination,name)
if __name__=='__main__':main()
