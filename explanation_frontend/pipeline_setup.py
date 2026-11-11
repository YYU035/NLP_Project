"""Explicit setup of pinned MIT NLI weights; inference never downloads files."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
NLI_SHA256 = "c03cd208bf920b4fbbb182a0535859a566e8acc3477d2b536bf87b769978524b"
ROOT = Path(__file__).resolve().parent
NLI_PATH = ROOT / ".models" / "claim-verifier"


def setup():
    NLI_PATH.mkdir(parents=True, exist_ok=True)
    names = ["README.md", "config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "spm.model", "model.safetensors"]
    for name in names:
        target = NLI_PATH / name
        if target.exists() and name != "model.safetensors":
            continue
        if target.exists():
            with target.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() == NLI_SHA256:
                    continue
        temporary = target.with_suffix(target.suffix + ".part")
        url = f"https://huggingface.co/{NLI_ID}/resolve/{NLI_REVISION}/{name}?download=true"
        total, reported = 0, 0
        with urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(4 * 1024 * 1024):
                handle.write(chunk)
                total += len(chunk)
                if total - reported > 50_000_000:
                    print(f"{name}: {total:,} bytes", flush=True)
                    reported = total
        if name == "model.safetensors":
            with temporary.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != NLI_SHA256:
                    raise ValueError("NLI weight checksum mismatch")
        temporary.replace(target)
        print(f"Installed {name}", flush=True)
    if "license: mit" not in (NLI_PATH / "README.md").read_text(encoding="utf-8").lower():
        raise ValueError("Expected MIT model card license")
    (NLI_PATH / "pinned_revision.txt").write_text(NLI_REVISION, encoding="utf-8")
    print(json.dumps({"model": NLI_ID, "revision": NLI_REVISION, "ready": True}), flush=True)


if __name__ == "__main__":
    setup()
