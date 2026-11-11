"""Explicit public download of pinned Apache-2.0 weights; never imported to infer."""
import os
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

from .llm import LOCAL_MODEL_ID, LOCAL_MODEL_REVISION


MODEL_WEIGHTS_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
MODEL_WEIGHTS_SIZE = 988097824


def download_weights(destination: Path):
    def checksum(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    if destination.is_file() and destination.stat().st_size == MODEL_WEIGHTS_SIZE and checksum(destination) == MODEL_WEIGHTS_SHA256:
        return
    request = Request(f"https://huggingface.co/{LOCAL_MODEL_ID}/resolve/{LOCAL_MODEL_REVISION}/model.safetensors?download=true")
    temporary = destination.with_suffix(".part")
    downloaded = 0
    last_report = 0
    with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        while chunk := response.read(4 * 1024 * 1024):
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_report >= 100_000_000:
                print(f"Public weights: {downloaded}/{MODEL_WEIGHTS_SIZE} bytes", flush=True)
                last_report = downloaded
    if downloaded != MODEL_WEIGHTS_SIZE or checksum(temporary) != MODEL_WEIGHTS_SHA256:
        raise ValueError("Public weights size/checksum mismatch; model was not activated")
    temporary.replace(destination)


def download_local_model(path: Path | None = None):
    root = Path(__file__).resolve().parent
    target = (path or root / ".models" / "Qwen2.5-0.5B-Instruct").resolve()
    if not target.is_relative_to(root):
        raise ValueError("Model setup must remain inside explanation_frontend")
    os.environ["HF_HOME"] = str(root / ".model-cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=LOCAL_MODEL_ID, revision=LOCAL_MODEL_REVISION, local_dir=str(target), token=False,
                      allow_patterns=["LICENSE", "README.md", "config.json", "generation_config.json",
                                      "tokenizer.json", "tokenizer_config.json",
                                      "vocab.json", "merges.txt"])
    download_weights(target / "model.safetensors")
    license_text = (target / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("Expected Apache-2.0 license is missing")
    (target / "pinned_revision.txt").write_text(LOCAL_MODEL_REVISION + "\n", encoding="utf-8")
    print(f"Downloaded {LOCAL_MODEL_ID} ({LOCAL_MODEL_REVISION}), Apache-2.0, to {target}")


if __name__ == "__main__":
    download_local_model()
