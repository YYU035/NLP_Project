"""Real local-model smoke check; constructed texts, not a retrieval benchmark."""

import importlib.metadata
import hashlib
import json
import platform
from pathlib import Path

from ranking_evaluation.examples.mock_input import build_mock_input
from ranking_evaluation.latency import measure_call
from ranking_evaluation.reranker import LocalCrossEncoder, rerank
from ranking_evaluation.rrf import reciprocal_rank_fusion
from ranking_evaluation.scripts.setup_reranker import MODEL_ID, MODEL_PATH, REVISION


if __name__ == "__main__":
    manifest = json.loads((MODEL_PATH / "download_manifest.json").read_text())
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise RuntimeError("Local model manifest does not match pinned revision")
    for name, expected in manifest["files"].items():
        with (MODEL_PATH / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise RuntimeError(f"Local model checksum mismatch: {name}")
    inputs = build_mock_input()
    claim = "How many people live in Berlin?"
    inputs.candidates.original_claim = claim
    texts = [
        "Bananas are a tropical fruit used in cooking.",
        "Berlin has a population of about 3.5 million people.",
        "Berlin has many art museums and historical buildings.",
    ]
    for entry, text in zip(inputs.candidates.papers, texts):
        entry.original_claim = claim
        entry.paper.title = text
    fused = reciprocal_rank_fusion(inputs)
    scorer, load_seconds = measure_call(lambda: LocalCrossEncoder(
        MODEL_PATH, device="cpu", max_length=512, batch_size=3, cpu_float64=True,
    ))
    cold, cold_seconds = measure_call(lambda: rerank(fused, claim=claim, scorer=scorer))
    warm, warm_seconds = measure_call(lambda: rerank(fused, claim=claim, scorer=scorer))
    if warm[0].entry.identity != "synthetic:B":
        raise RuntimeError("Expected population passage to rank first")
    report = {
        "type": "real CPU inference smoke check on constructed texts; not benchmark",
        "model_id": MODEL_ID, "revision": REVISION,
        "weight_manifest": manifest,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "sentence-transformers", "transformers", "torch", "huggingface-hub",
        )},
        "device": "cpu", "dtype": "float64", "max_length": 512, "batch_size": 3, "top_n": 20,
        "load_seconds": load_seconds, "cold_rerank_seconds": cold_seconds,
        "warm_rerank_seconds": warm_seconds,
        "papers": [{"identity": row.entry.identity, "score": row.cross_encoder_score}
                   for row in warm],
        "cold_order": [row.entry.identity for row in cold],
    }
    output = Path("outputs/ranking_evaluation/cpu_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
