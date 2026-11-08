"""Explicit download of the two versioned PDFs used for separate manual review."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES=[('transformer','1706.03762v7','bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697'),
         ('bert','1810.04805v2','5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e')]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name,version,expected in SOURCES:
        path=a.output_dir/(name+'.pdf');url='https://arxiv.org/pdf/'+version
        urllib.request.urlretrieve(url,path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=expected:raise ValueError('Pinned PDF bytes differ; inspect before updating hashes: '+version)
        rows.append({'paper_id':'arxiv:'+version,'url':url,'file':str(path.resolve()),'sha256':digest})
    (a.output_dir/'sources.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(a.output_dir/'sources.json')


if __name__=='__main__':main()
