"""Explicit bounded downloads of pinned official dataset files."""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

SOURCES = {
    "reciper": {
        "repository": "ReaganWZY/RECIPER",
        "revision": "1e167badbf19d6fd78b2a49700028f4c68134219",
        "host": "huggingface",
        "files": ["README.md", "LICENSE", "data/rag_database/qa_dataset.jsonl",
                  "data/rag_database/rag_database.json"],
    },
    "climatecheck": {
        "repository": "rabuahmad/climatecheck",
        "revision": "93d0dc5007e9f86a3fc4cfe2d7998514a94f5c88",
        "host": "huggingface",
        "files": ["README.md", "data/train-00000-of-00001.parquet",
                  "data/test-00000-of-00001.parquet"],
    },
    "csfcube": {
        "repository": "iesl/CSFCube",
        "revision": "7ffe012c1d92c470f185fc086086caf5b8e05ef3",
        "files": ["README.md", "LICENSE.md", "datasheet.md",
                  "abstracts-csfcube-preds.jsonl", "evaluation_splits.json",
                  *[f"test-pid2anns-csfcube-{facet}.json"
                    for facet in ("background", "method", "result")]],
    },
    "matter_of_fact": {
        "repository": "cognitiveailab/matter-of-fact",
        "revision": "a012d8182fbf25434c4389102e81ad282528a100",
        "files": ["README.md", "benchmark/matteroffact.train.2022.1376.json",
                  "benchmark/matteroffact.validation.2023.2538.json",
                  "benchmark/matteroffact.test.20242025.4446.json"],
    },
}


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_download(directory: Path, dataset: str) -> dict:
    source = SOURCES[dataset]
    manifest = json.loads((directory / "download_manifest.json").read_text(encoding='utf-8'))
    if manifest["repository"] != source["repository"] or manifest["revision"] != source["revision"]:
        raise ValueError("Dataset manifest differs from the pinned source")
    if set(manifest["files_sha256"]) != set(source["files"]):
        raise ValueError("Dataset manifest file set is incomplete or unexpected")
    for name, expected in manifest["files_sha256"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    return manifest


def download(dataset: str, directory: Path) -> dict:
    """Reuse only verified completed downloads; never replace existing data."""
    source = SOURCES[dataset]
    if directory.exists():
        return verify_download(directory, dataset)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "payload"
        staging.mkdir()
        hashes = {}
        for name in source["files"]:
            if source.get("host") == "huggingface":
                url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{name}"
            else:
                url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{name}"
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": "NLP-Project-dataset-setup"})
            with urllib.request.urlopen(request, timeout=60) as response, target.open("xb") as handle:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 50 * 1024 * 1024:
                        raise ValueError("Dataset file exceeds the 50 MiB download budget")
                    handle.write(chunk)
            hashes[name] = file_hash(target)
        manifest = {"dataset": dataset, "repository": source["repository"],
                    "revision": source["revision"], "files_sha256": hashes}
        (staging / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if directory.exists():
            raise FileExistsError(directory)
        shutil.move(str(staging), str(directory))
    return verify_download(directory, dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=list(SOURCES))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    directory = args.data_root / args.dataset
    manifest = download(args.dataset, directory)
    print(json.dumps({"path": str(directory), "revision": manifest["revision"]}, indent=2))


if __name__ == "__main__":
    main()
