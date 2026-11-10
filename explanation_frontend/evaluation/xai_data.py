"""Stage exact public SciFact data already present in pinned Git history; no download."""
import hashlib
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REVISION='7ed93fa4be69c661422b33b6c88b949dd2636e12'
EXPECTED={'corpus.jsonl':'b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62',
          'claims_dev.jsonl':'86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217'}


def main():
    target=ROOT/'outputs/xai/data'
    target.mkdir(parents=True,exist_ok=True)
    for name,digest in EXPECTED.items():
        data=subprocess.check_output(['git','show',f'{REVISION}:claim_verifier/data/{name}'],cwd=ROOT)
        if hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError(f'Unexpected pinned SciFact bytes: {name}')
        (target/name).write_bytes(data)
    print(f'Staged verified SciFact data: {target}')

if __name__=='__main__':main()
