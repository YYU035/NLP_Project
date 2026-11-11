"""Public-data setup must preserve exact inputs without extracting arbitrary paths."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from evaluation import datasets


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def make_release(tmp_path, monkeypatch, *, extra=None, omit=None):
    files = {"corpus.jsonl": b'{"doc_id":1}\n',
             "claims_dev.jsonl": b'{"id":2}\n',
             "claims_train.jsonl": b'{"id":3}\n'}
    path = tmp_path / "release.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in files.items():
            if name == omit:
                continue
            member = tarfile.TarInfo("data/" + name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        if extra is not None:
            member, raw = extra
            archive.addfile(member, io.BytesIO(raw))
    monkeypatch.setattr(datasets, "SCIFACT_SHA256", {name: digest(raw) for name, raw in files.items()})
    monkeypatch.setattr(datasets, "SCIFACT_ARCHIVE_SHA256", digest(path.read_bytes()))
    return path, files


def test_explicit_download_is_verified_and_cached_without_further_network(tmp_path, monkeypatch):
    archive, files = make_release(tmp_path, monkeypatch)
    calls = []
    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(archive.read_bytes())
    monkeypatch.setattr(datasets, "urlopen", opener)
    destination = tmp_path / "staged"
    datasets.stage_scifact(destination)
    first = {name: (destination / name).read_bytes() for name in files}
    datasets.stage_scifact(destination)
    assert first == files
    assert calls == [(datasets.SCIFACT_URL, 60)]
    assert set(path.name for path in destination.iterdir()) == {*files, "download_manifest.json"}


def test_loading_missing_or_corrupt_inputs_never_downloads_or_replaces_them(tmp_path, monkeypatch):
    archive, _ = make_release(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only loading cannot download")
    monkeypatch.setattr(datasets, "urlopen", forbidden)
    with pytest.raises(FileNotFoundError, match="Stage public data first"):
        datasets.read_scifact_file("corpus.jsonl", tmp_path / "missing")
    destination = tmp_path / "staged"
    datasets.stage_scifact(destination, archive_path=archive)
    target = destination / "corpus.jsonl"
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        datasets.stage_scifact(destination)
    assert target.read_bytes() == b"corrupt"
    with pytest.raises(ValueError, match="Unknown"):
        datasets.read_scifact_file("../private.json", destination)


@pytest.mark.parametrize("name", ["../../outside", "/absolute", "data\\corpus.jsonl"])
def test_archive_paths_cannot_escape_the_named_payload(tmp_path, monkeypatch, name):
    member = tarfile.TarInfo(name)
    member.size = 1
    archive, _ = make_release(tmp_path, monkeypatch, extra=(member, b"x"))
    destination = tmp_path / "staged"
    with pytest.raises(ValueError, match="Unsafe path"):
        datasets.stage_scifact(destination, archive_path=archive)
    assert not destination.exists()


def test_links_duplicates_and_missing_members_are_rejected_before_publication(tmp_path, monkeypatch):
    link = tarfile.TarInfo("data/corpus.jsonl")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../private"
    archive, _ = make_release(tmp_path, monkeypatch, extra=(link, b""))
    destination = tmp_path / "staged"
    with pytest.raises(ValueError, match="regular file"):
        datasets.stage_scifact(destination, archive_path=archive)
    assert not destination.exists()
    duplicate = tarfile.TarInfo("data/corpus.jsonl")
    duplicate.size = 1
    archive, _ = make_release(tmp_path, monkeypatch, extra=(duplicate, b"x"))
    with pytest.raises(ValueError, match="Duplicate"):
        datasets.stage_scifact(destination, archive_path=archive)
    archive, _ = make_release(tmp_path, monkeypatch, omit="claims_train.jsonl")
    with pytest.raises(ValueError, match="missing required"):
        datasets.stage_scifact(destination, archive_path=archive)
    assert not destination.exists()


def test_archive_hash_and_expansion_budgets_are_enforced(tmp_path, monkeypatch):
    archive, _ = make_release(tmp_path, monkeypatch)
    destination = tmp_path / "staged"
    monkeypatch.setattr(datasets, "SCIFACT_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="archive checksum"):
        datasets.stage_scifact(destination, archive_path=archive)
    monkeypatch.setattr(datasets, "SCIFACT_ARCHIVE_SHA256", digest(archive.read_bytes()))
    monkeypatch.setattr(datasets, "MAX_MEMBER_BYTES", 1)
    with pytest.raises(ValueError, match="expanded-byte budget"):
        datasets.stage_scifact(destination, archive_path=archive)
    assert not destination.exists()


def test_historical_prediction_bundle_detects_changed_compressed_bytes(tmp_path, monkeypatch):
    raw = b'{"claim_id":"1","prediction":"SUPPORTED"}\n'
    compressed = gzip.compress(raw, mtime=0)
    relative = "evaluation/reference_predictions/context_train.jsonl.gz"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(compressed)
    manifest = tmp_path / "docs/contributions/evaluation_sources.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"reference_predictions": {"context_train": {
        "path": relative, "raw_bytes": len(raw), "raw_sha256": digest(raw),
        "gzip_sha256": digest(compressed)}}}), encoding="utf-8")
    monkeypatch.setattr(datasets, "PROJECT_ROOT", tmp_path)
    assert datasets.read_reference_predictions("context_train") == raw
    path.write_bytes(compressed + b"tampered")
    with pytest.raises(ValueError, match="compressed checksum"):
        datasets.read_reference_predictions("context_train")


def test_bundled_training_and_development_references_retain_original_cohorts():
    training = [json.loads(row) for row in datasets.read_reference_predictions("context_train").splitlines()]
    arms = {policy: {row["claim_id"] for row in training if row["policy"] == policy and row["depth"] == 5}
            for policy in ["baseline", "bounded_abstract"]}
    assert len(arms["baseline"]) == 200
    assert arms["baseline"] == arms["bounded_abstract"]
    development = [json.loads(row) for row in datasets.read_reference_predictions("context_dev").splitlines()]
    selected = [row for row in development if row["depth"] == 5 and row["condition"] == "train_selected"]
    assert len(selected) == 300 and len({row["claim_id"] for row in selected}) == 300
