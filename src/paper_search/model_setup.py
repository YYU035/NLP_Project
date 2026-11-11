"""Explicit setup of the existing pinned model; never run on import."""

import argparse
import os
from pathlib import Path

from paper_search.keybert_queries import (
    DEFAULT_MODEL_PATH,
    MODEL_ID,
    MODEL_REVISION,
)


def download_model(path: Path) -> None:
    """Download pinned local weights, preserving the experiment setup logic."""
    os.environ["HF_HOME"] = str(path.resolve().parent / "hf-cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(path),
        allow_patterns=[
            "config.json",
            "modules.json",
            "model.safetensors",
            "pytorch_model.bin",
            "sentence_bert_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "1_Pooling/config.json",
            "README.md",
        ],
    )
    (path / "pinned_revision.txt").write_text(MODEL_REVISION, encoding="utf-8")


def main() -> None:
    """Make setup independent of experimental scripts and retrieval calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    download_model(args.model_path)


if __name__ == "__main__":
    main()
