"""Run declared RRF/re-ranking ablations against one immutable input snapshot."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ranking_evaluation.benchmark import BenchmarkBundle

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ks: list[PositiveInt] = Field(default=[5, 10, 20], min_length=1)
    rrf_constants: list[NonNegativeInt] = Field(default=[20, 60, 100], min_length=1)
    rerank_depths: list[PositiveInt] = Field(default_factory=list)
    repeats: PositiveInt = 1

    @model_validator(mode="after")
    def unique_values(self):
        for name in ("ks", "rrf_constants", "rerank_depths"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
        return self


def summarize_report(report: dict, methods: list[str], *, run: str, repeat: int) -> list[dict]:
    """Summarize measured stages only; never sum concurrent/replayed timings."""
    rows = []
    for method in methods:
        metrics = report["metrics"][method]
        row = {
            "run": run, "repeat": repeat, "method": method,
            "synthetic": report["dataset"]["synthetic"],
            "rrf_k": report["configuration"]["rrf_k"] if method.startswith("rrf") else None,
            "top_n": report["configuration"]["top_n"] if method == "rrf_cross_encoder" else None,
            "evaluated_claims": metrics["evaluated_claims"],
            "excluded_no_relevant": metrics["excluded_no_relevant"],
            **metrics["macro"],
        }
        stages = {"bm25": "bm25_seconds", "sbert": "sbert_seconds",
                  "rrf": "rrf_seconds", "rrf_cross_encoder": "rerank_seconds"}
        field = stages[method]
        values = sorted(timing[field] for timing in report["latency"].values()
                        if timing[field] is not None)
        row.update(
            latency_stage=field, latency_observations=len(values),
            latency_mean_seconds=mean(values) if values else None,
            latency_median_seconds=median(values) if values else None,
            latency_p95_seconds=values[math.ceil(0.95 * len(values)) - 1] if values else None,
        )
        rows.append(row)
    return rows


def run_suite(
    input_path: Path, config: SuiteConfig, output_dir: Path,
    *, model_options: list[str] | None = None,
) -> dict:
    """Execute evaluation CLI per run; each model run loads its own model.

    Repeats replay exported rankings, not retrieval inference. The first model
    inference remains cold. No best configuration is automatically selected.
    """
    config = SuiteConfig.model_validate(config.model_dump())
    raw = input_path.read_bytes()
    bundle = BenchmarkBundle.model_validate_json(raw)
    if config.rerank_depths and not model_options:
        raise ValueError("Rerank depths require explicit local model options")
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = output_dir / "input.snapshot.json"
    snapshot.write_bytes(raw)
    (output_dir / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    manifest = {
        "status": "running", "synthetic": bundle.synthetic,
        "dataset_name": bundle.dataset_name, "split": bundle.split,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol": "paired fixed-input replay; no retrieval inference or automatic model selection",
        "latency_note": "p95 uses nearest rank across claims per stage; no stage sums; rerank timing includes cold inference; one repeat is a pilot, not stable latency evidence",
        "config": config.model_dump(), "runs": [],
    }
    manifest_path = output_dir / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "live_search/src"))
    rows = []
    save_manifest()
    try:
        for repeat in range(1, config.repeats + 1):
            for index, constant in enumerate(config.rrf_constants):
                jobs = [(None, ["bm25", "sbert", "rrf"] if index == 0 else ["rrf"])]
                jobs += [(depth, ["rrf_cross_encoder"]) for depth in config.rerank_depths]
                for depth, methods in jobs:
                    name = f"repeat-{repeat}-rrf-{constant}"
                    if depth is not None:
                        name += f"-rerank-{depth}"
                    destination = output_dir / f"{name}.json"
                    command = [sys.executable, "-m", "ranking_evaluation.evaluate",
                               "--input", str(snapshot.resolve()),
                               "--output", str(destination.resolve()),
                               "--rrf-k", str(constant), "--ks", *map(str, config.ks)]
                    if depth is not None:
                        command += ["--top-n", str(depth), *model_options]
                    completed = subprocess.run(command, cwd=root, env=env,
                                               capture_output=True, text=True)
                    (output_dir / f"{name}.log").write_text(
                        completed.stdout + completed.stderr, encoding="utf-8",
                    )
                    if completed.returncode:
                        raise RuntimeError(f"Run {name} failed; inspect {name}.log")
                    report = json.loads(destination.read_text())
                    if report["provenance"]["input_sha256"] != manifest["input_sha256"]:
                        raise RuntimeError("Evaluation input hash changed")
                    rows.extend(summarize_report(report, methods, run=name, repeat=repeat))
                    manifest["runs"].append({"name": name, "report": destination.name,
                                             "report_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                                             "methods_in_summary": methods})
                    save_manifest()
        (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save_manifest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-float64", action="store_true")
    args = parser.parse_args()
    if args.cpu_float64 and args.device != "cpu":
        parser.error("--cpu-float64 requires --device cpu")
    config = SuiteConfig.model_validate_json(args.config.read_bytes())
    options = None
    if args.model_path:
        if not args.model_id or not args.model_revision:
            parser.error("Model ID and revision are required")
        options = ["--model-path", str(args.model_path.resolve()),
                   "--model-id", args.model_id, "--model-revision", args.model_revision,
                   "--batch-size", str(args.batch_size), "--max-length", str(args.max_length),
                   "--device", args.device]
        if args.cpu_float64:
            options.append("--cpu-float64")
    elif args.model_id or args.model_revision or args.cpu_float64 or args.device != "cpu":
        parser.error("Model options require --model-path")
    run_suite(args.input, config, args.output_dir, model_options=options)
    print(args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
