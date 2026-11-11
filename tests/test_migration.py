"""Release gates protect original evidence while package paths change."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.check_integrity import (
    IntegrityError, check_migration, check_release, check_source_hygiene,
    historical_path, safe_path,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_preserves_all_frozen_evidence_and_examples():
    report = check_release(ROOT)
    assert report["migration_files"] > 400
    assert report["preserved_files"] > 200
    assert report["frozen_manifests"] == 4
    assert report["frozen_artifacts"] > 100
    assert report["benchmark_artifacts"] == 21
    assert report["demo_replays"] == 8
    assert report["recorded_attributions"] >= 1


@pytest.mark.parametrize("name", [
    "../secret", "a/../../secret", "a\\..\\secret", "/secret",
    "C:/secret", "C:secret", "\\\\server\\share\\secret", "", ".",
])
def test_manifest_paths_cannot_escape_the_release_root(tmp_path, name):
    with pytest.raises(IntegrityError):
        safe_path(tmp_path, name)


def test_preserved_evidence_and_transformed_source_have_separate_hashes(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"threshold = 0.5\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {"old_path": "old.py", "new_path": "source.py",
             "original_sha256": digest, "preserve_bytes": True}
    assert check_migration(tmp_path, {"files": [entry]})["preserved_files"] == 1
    path.write_bytes(b"threshold = 0.9\n")
    with pytest.raises(IntegrityError, match="Checksum mismatch"):
        check_migration(tmp_path, {"files": [entry]})
    entry.update(preserve_bytes=False,
                 migrated_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert check_migration(tmp_path, {"files": [entry]})["transformed_files"] == 1
    path.write_bytes(b"threshold = 0.1\n")
    with pytest.raises(IntegrityError, match="Checksum mismatch"):
        check_migration(tmp_path, {"files": [entry]})


def test_missing_destination_and_duplicate_mapping_are_rejected(tmp_path):
    entry = {"old_path": "old.json", "new_path": "result.json",
             "original_sha256": "0" * 64, "preserve_bytes": False}
    with pytest.raises(IntegrityError, match="Missing migration destination"):
        check_migration(tmp_path, {"files": [entry]})
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(IntegrityError, match="Duplicate migration path"):
        check_migration(tmp_path, {"files": [entry, entry]})


def test_historical_windows_names_resolve_through_path_map(tmp_path):
    mapped = {"explanation_frontend/evaluation/study/result.json":
              "evaluation/frozen_study/result.json"}
    path = historical_path(tmp_path, "evaluation\\study\\result.json", mapped)
    assert path == tmp_path / "evaluation/frozen_study/result.json"
    with pytest.raises(IntegrityError):
        historical_path(tmp_path, "..\\secret", mapped)


def test_source_tree_rejects_model_weights_and_old_contribution_folders(tmp_path):
    source = tmp_path / "src/research_verifier"
    source.mkdir(parents=True)
    (source / "weights.safetensors").write_bytes(b"fake weights")
    with pytest.raises(IntegrityError, match="Model weights"):
        check_source_hygiene(tmp_path)
    (source / "weights.safetensors").unlink()
    (tmp_path / "live_search").mkdir()
    with pytest.raises(IntegrityError, match="Legacy contribution"):
        check_source_hygiene(tmp_path)
