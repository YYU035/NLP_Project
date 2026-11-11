"""Explicit, bounded public-data setup and verified historical prediction inputs.

Run ``python -m evaluation.datasets scifact`` before model-based evaluations.
Loading a dataset never downloads it and never reads another Git branch.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

from research_verifier.runtime.paths import PROJECT_ROOT

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_ARCHIVE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
SCIFACT_SHA256 = {
    "corpus.jsonl": "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62",
    "claims_dev.jsonl": "86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217",
    "claims_train.jsonl": "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0",
}
SCIFACT_DIRECTORY = PROJECT_ROOT / "outputs/xai/data"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_scifact_file(name: str, directory: Path | None = None) -> bytes:
    """Read only an explicitly staged, exact release file; no network fallback."""
    if name not in SCIFACT_SHA256:
        raise ValueError("Unknown pinned SciFact file")
    path = (directory or SCIFACT_DIRECTORY) / name
    if path.is_symlink():
        raise ValueError("SciFact input cannot be a symbolic link")
    if not path.is_file():
        raise FileNotFoundError("Stage public data first: python -m evaluation.datasets scifact")
    raw = path.read_bytes()
    if sha256(raw) != SCIFACT_SHA256[name]:
        raise ValueError("SciFact checksum mismatch: " + name)
    return raw


def _extract_release(archive_path: Path, destination: Path) -> None:
    """Extract only three named regular files, without tar path extraction."""
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("SciFact archive exceeds compressed-byte budget")
    if sha256(archive_path.read_bytes()) != SCIFACT_ARCHIVE_SHA256:
        raise ValueError("SciFact archive checksum mismatch")
    found = set()
    expanded = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Unsafe path in SciFact archive")
            expanded += member.size
            if member.size > MAX_MEMBER_BYTES or expanded > MAX_EXPANDED_BYTES:
                raise ValueError("SciFact archive exceeds expanded-byte budget")
            name = path.name
            if member.name not in {"data/" + value for value in SCIFACT_SHA256}:
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("SciFact release input must be a regular file")
            if name in found:
                raise ValueError("Duplicate SciFact release member")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Missing SciFact release member body")
            with handle:
                raw = handle.read(MAX_MEMBER_BYTES + 1)
            if len(raw) != member.size or sha256(raw) != SCIFACT_SHA256[name]:
                raise ValueError("SciFact member checksum mismatch: " + name)
            (destination / name).write_bytes(raw)
            found.add(name)
    if found != set(SCIFACT_SHA256):
        raise ValueError("SciFact archive is missing required files")


def stage_scifact(directory: Path | None = None, *, archive_path: Path | None = None) -> dict:
    """Explicitly download or reuse exact public release data without replacing inputs."""
    directory = Path(directory or SCIFACT_DIRECTORY)
    existing = {name for name in SCIFACT_SHA256 if (directory / name).exists()}
    for name in existing:
        read_scifact_file(name, directory)
    if existing != set(SCIFACT_SHA256):
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".scifact-", dir=directory.parent) as temporary:
            staging = Path(temporary)
            if archive_path is None:
                archive_path = staging / "data.tar.gz"
                request = Request(SCIFACT_URL, headers={"User-Agent": "research-verifier-data-setup"})
                with urlopen(request, timeout=60) as response, archive_path.open("xb") as output:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_ARCHIVE_BYTES:
                            raise ValueError("SciFact download exceeds compressed-byte budget")
                        output.write(chunk)
            payload = staging / "payload"
            payload.mkdir()
            _extract_release(Path(archive_path), payload)
            directory.mkdir(parents=True, exist_ok=True)
            for name in SCIFACT_SHA256:
                if name in existing:
                    continue
                with (payload / name).open("rb") as source, (directory / name).open("xb") as target:
                    shutil.copyfileobj(source, target)
    for name in SCIFACT_SHA256:
        read_scifact_file(name, directory)
    manifest = {"url": SCIFACT_URL, "archive_sha256": SCIFACT_ARCHIVE_SHA256,
                "files_sha256": SCIFACT_SHA256}
    manifest_path = directory / "download_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing SciFact manifest differs from pinned release")
    else:
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    return manifest


def read_reference_predictions(name: str) -> bytes:
    """Load bundled measured owner predictions, checking both compression and raw bytes."""
    manifest = json.loads((PROJECT_ROOT / "docs/contributions/evaluation_sources.json").read_text(encoding="utf-8"))
    try:
        source = manifest["reference_predictions"][name]
    except KeyError as error:
        raise ValueError("Unknown historical prediction reference") from error
    path = (PROJECT_ROOT / source["path"]).resolve()
    if not path.is_relative_to((PROJECT_ROOT / "evaluation/reference_predictions").resolve()):
        raise ValueError("Prediction reference path escapes its bundle")
    compressed = path.read_bytes()
    if sha256(compressed) != source["gzip_sha256"]:
        raise ValueError("Prediction reference compressed checksum mismatch")
    with gzip.open(path, "rb") as handle:
        raw = handle.read(source["raw_bytes"] + 1)
    if len(raw) != source["raw_bytes"] or sha256(raw) != source["raw_sha256"]:
        raise ValueError("Prediction reference raw checksum mismatch")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["scifact"])
    parser.add_argument("--directory", type=Path, default=SCIFACT_DIRECTORY)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded exact release archive")
    args = parser.parse_args()
    stage_scifact(args.directory, archive_path=args.archive)
    print("Verified public SciFact data:", args.directory)


if __name__ == "__main__":
    main()
