"""Explicit fixed-revision download; never imported by the ranking path."""

import hashlib
import json
from pathlib import Path

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MODEL_PATH = Path(__file__).resolve().parents[2] / ".cache" / "ranking" / "ms-marco-MiniLM-L6-v2"
FILES = ["config.json", "model.safetensors", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.txt", "README.md"]


if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_ID, revision=REVISION,
                      local_dir=MODEL_PATH, allow_patterns=FILES)
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "files": {}}
    for name in FILES:
        with (MODEL_PATH / name).open("rb") as handle:
            manifest["files"][name] = hashlib.file_digest(handle, "sha256").hexdigest()
    (MODEL_PATH / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(MODEL_PATH)
