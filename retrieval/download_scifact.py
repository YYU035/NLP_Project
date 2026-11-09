"""Download the official SciFact release into data/scifact/.

    python -m retrieval.download_scifact

Files: corpus.jsonl (5,183 abstracts), claims_train.jsonl, claims_dev.jsonl,
claims_test.jsonl (test has no labels). If the shared/ folder ends up owning
the dataset, point --data-dir at that copy instead and skip this script.
"""
import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scifact")
    out = Path(ap.parse_args().out)
    if (out / "corpus.jsonl").exists():
        print(f"Already present: {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "data.tar.gz"
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(tmp, filter="data")
        src = Path(tmp) / "data"
        for f in src.glob("*.jsonl"):
            shutil.copy(f, out / f.name)
    print("Saved:", sorted(p.name for p in out.glob("*.jsonl")))


if __name__ == "__main__":
    main()
