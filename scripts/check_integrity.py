"""Verify release provenance and bundled evidence without Git objects or network.

Run from any directory: python /path/to/project/scripts/check_integrity.py
Local environments, model/data downloads and runtime caches are intentionally
outside the proposed source tree. An optional index-only check also catches
tracked generated files; neither check reads historical Git objects.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIRECTORIES = (
    "explanation_frontend", "live_search", "retrieval", "ranking_evaluation",
    "evidence_verification", "shared",
)
SOURCE_DIRECTORIES = (
    "src", "tests", "scripts", "config", "requirements", "examples",
    "evaluation", "docs", ".github", ".streamlit",
)
GENERATED_DIRECTORIES = {
    "__pycache__", ".pytest_cache", ".cache", ".tmp", ".venv", ".pip-cache",
}
MODEL_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".onnx", ".h5", ".ckpt"}
SHA256 = re.compile(r"[0-9a-f]{64}")


class IntegrityError(ValueError):
    """A release file or manifest violates an integrity invariant."""


def normalized_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise IntegrityError("Manifest path must be a nonempty string")
    windows = PureWindowsPath(name)
    value = name.replace("\\", "/")
    posix = PurePosixPath(value)
    if windows.drive or windows.root or posix.is_absolute() or ".." in posix.parts:
        raise IntegrityError(f"Manifest path escapes its root: {name}")
    if str(posix) == ".":
        raise IntegrityError("Manifest path must name a file")
    return str(posix)


def safe_path(root: Path, name: str) -> Path:
    root = root.resolve()
    path = (root / normalized_name(name)).resolve()
    if not path.is_relative_to(root):
        raise IntegrityError(f"Manifest path escapes its root: {name}")
    return path


def file_hash(path: Path) -> str:
    if not path.is_file():
        raise IntegrityError(f"Missing release file: {path}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        raise IntegrityError(f"Invalid SHA-256 for {path}")
    if file_hash(path) != expected:
        raise IntegrityError(f"Checksum mismatch: {path}")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Cannot read manifest: {path}") from exc


def check_migration(root: Path, manifest: dict) -> dict:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise IntegrityError("Migration manifest has no file entries")
    seen_old, seen_new = set(), set()
    preserved = transformed = unfinalized = 0
    for entry in entries:
        old = normalized_name(entry["old_path"])
        new = normalized_name(entry["new_path"])
        if old in seen_old or new in seen_new:
            raise IntegrityError(f"Duplicate migration path: {old} -> {new}")
        seen_old.add(old)
        seen_new.add(new)
        path = safe_path(root, new)
        if not path.is_file():
            raise IntegrityError(f"Missing migration destination: {new}")
        if entry.get("preserve_bytes") is True:
            verify_hash(path, entry["original_sha256"])
            preserved += 1
        elif entry.get("preserve_bytes") is False:
            if entry.get("migrated_sha256"):
                verify_hash(path, entry["migrated_sha256"])
                transformed += 1
            else:
                unfinalized += 1
        else:
            raise IntegrityError(f"Missing preservation policy: {new}")
    return {"migration_files": len(entries), "preserved_files": preserved,
            "transformed_files": transformed, "unfinalized_files": unfinalized}


def check_hash_mapping(root: Path, mapping: dict) -> int:
    if not isinstance(mapping, dict) or not mapping:
        raise IntegrityError(f"Empty artifact hash mapping: {root}")
    for name, expected in mapping.items():
        verify_hash(safe_path(root, name), expected)
    return len(mapping)


def historical_path(root: Path, name: str, migrated_paths: dict[str, str]) -> Path:
    name = normalized_name(name)
    # Original ARTIFACTS manifests used the old application's root and often
    # Windows separators. Keep their bytes and resolve their names explicitly.
    destination = migrated_paths.get(name)
    if destination is None:
        destination = migrated_paths.get("explanation_frontend/" + name, name)
    return safe_path(root, destination)


def check_frozen_artifacts(root: Path, migration: dict) -> dict:
    paths = {normalized_name(e["old_path"]): normalized_name(e["new_path"])
             for e in migration["files"]}
    manifests = sorted((root / "evaluation").rglob("ARTIFACTS.json"))
    if not manifests:
        raise IntegrityError("No frozen evaluation artifact manifests found")
    total = 0
    for manifest in manifests:
        hashes = read_json(manifest)
        if not isinstance(hashes, dict) or not hashes:
            raise IntegrityError(f"Empty frozen artifact manifest: {manifest}")
        for name, expected in hashes.items():
            verify_hash(historical_path(root, name, paths), expected)
            total += 1
    return {"frozen_manifests": len(manifests), "frozen_artifacts": total}


def check_examples(root: Path) -> dict:
    benchmarks = root / "examples/benchmarks"
    benchmark_count = check_hash_mapping(
        benchmarks, read_json(benchmarks / "SOURCE.json")["sha256"])
    replays = root / "examples/replays"
    replay_count = check_hash_mapping(
        replays, read_json(replays / "manifest.json")["sha256"])
    attributions = replays / "attributions"
    recorded = read_json(attributions / "manifest.json")
    if not isinstance(recorded, dict) or not recorded:
        raise IntegrityError("No recorded attribution entries")
    for item in recorded.values():
        verify_hash(safe_path(attributions, item["file"]), item["sha256"])
    return {"benchmark_artifacts": benchmark_count, "demo_replays": replay_count,
            "recorded_attributions": len(recorded)}


def _forbidden_release_file(relative: Path) -> bool:
    return (relative.suffix.lower() in MODEL_SUFFIXES
            or relative.suffix.lower() in {".pyc", ".pyo"}
            or any(part in GENERATED_DIRECTORIES or part.endswith(".egg-info")
                   for part in relative.parts))


def check_source_hygiene(root: Path) -> dict:
    for name in LEGACY_DIRECTORIES:
        if (root / name).exists():
            raise IntegrityError(f"Legacy contribution directory remains: {name}")
    scanned = 0
    for name in SOURCE_DIRECTORIES:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            relative = path.relative_to(root)
            # Editable installs/imports can create ignored local caches. These
            # are not source artifacts; a tracked instance is rejected below.
            if any(part in GENERATED_DIRECTORIES or part.endswith(".egg-info")
                   for part in relative.parts):
                continue
            if path.is_file():
                scanned += 1
                if path.suffix.lower() in MODEL_SUFFIXES:
                    raise IntegrityError(f"Model weights in source tree: {relative}")
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True,
            check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        result = None
    # This only needs a working index, not branch history. Source ZIPs can use
    # the filesystem/provenance gates without Git installed.
    tracked = 0
    if result is not None and result.returncode == 0:
        for item in result.stdout.decode("utf-8").split("\x00"):
            if not item:
                continue
            relative = Path(normalized_name(item))
            if (root / relative).is_file():
                tracked += 1
                if _forbidden_release_file(relative):
                    raise IntegrityError(f"Generated file tracked in source: {relative}")
    return {"source_files_scanned": scanned, "tracked_files_checked": tracked}


def check_release(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifest = read_json(root / "docs/architecture/migration_manifest.json")
    migration = check_migration(root, manifest)
    if migration["unfinalized_files"]:
        raise IntegrityError("Migration source hashes have not been finalized")
    return {"status": "pass", **migration,
            **check_frozen_artifacts(root, manifest), **check_examples(root),
            **check_source_hygiene(root)}


def main() -> int:
    try:
        result = check_release()
    except (IntegrityError, KeyError, TypeError) as exc:
        print(f"Integrity check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
