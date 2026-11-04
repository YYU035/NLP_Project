"""Explicit official dataset download; pin obtained bytes with SHA-256."""

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "data" / "scifact"


if __name__ == "__main__":
    if DESTINATION.exists():
        raise SystemExit("Destination exists; preserve it and choose a separate version manually")
    cache = ROOT / ".cache" / "ranking"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "scifact-data.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    names = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl")
    contents = {}
    with tarfile.open(archive, "r:gz") as source:
        for name in names:
            matches = [member for member in source.getmembers()
                       if Path(member.name).name == name and member.isfile()]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one archive entry: {name}")
            with source.extractfile(matches[0]) as handle:
                contents[name] = handle.read()
    # Only explicit filenames are written; never extract arbitrary archive paths.
    DESTINATION.mkdir(parents=True)
    for name, content in contents.items():
        (DESTINATION / name).write_bytes(content)
    with archive.open("rb") as handle:
        archive_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "url": URL, "source_note": "latest URL is mutable; hashes pin the downloaded bytes",
        "archive_sha256": archive_hash,
        "files_sha256": {name: hashlib.sha256(content).hexdigest()
                         for name, content in contents.items()},
    }
    (DESTINATION / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(DESTINATION)
