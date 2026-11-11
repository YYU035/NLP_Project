"""Explicit project commands; launching the app never provisions models."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from research_verifier.runtime.paths import APP_PATH, DATA_DIR, PROJECT_ROOT, RETRIEVAL_MODEL_DIR


def _run(arguments: list[str]) -> int:
    env = os.environ.copy()
    # Paths were resolved against the caller's cwd. Freeze those absolute
    # values before changing the child process to the resource checkout.
    env["RESEARCH_VERIFIER_ROOT"] = str(PROJECT_ROOT)
    env["RESEARCH_VERIFIER_DATA_DIR"] = str(DATA_DIR)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    return subprocess.run([sys.executable, *arguments], cwd=PROJECT_ROOT, env=env).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scientific claim verification workspace")
    commands = parser.add_subparsers(dest="command", required=True)

    launch = commands.add_parser("run", help="Launch the local Streamlit workspace")
    launch.add_argument("--port", type=int, default=8501)

    smoke = commands.add_parser("smoke", help="Run a recorded or explicit live model smoke")
    smoke.add_argument("--live", action="store_true", help="Call public discovery catalogues")
    smoke.add_argument("--baseline", action="store_true")
    smoke.add_argument("--limit", type=int, default=3)
    smoke.add_argument("--claim")

    models = commands.add_parser("setup-models", help="Explicitly download pinned public models")
    models.add_argument("--language-model", action="store_true", help="Also provision experimental local Qwen")
    commands.add_parser("check", help="Verify the preserved research and example artifacts")
    args = parser.parse_args(argv)

    if args.command == "run":
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        return _run(["-m", "streamlit", "run", str(APP_PATH), "--server.address", "127.0.0.1",
                     "--server.port", str(args.port)])
    if args.command == "smoke":
        if not 1 <= args.limit <= 10:
            parser.error("--limit must be between 1 and 10")
        flags = ["--limit", str(args.limit)]
        if args.live:
            flags.append("--live")
        if args.baseline:
            flags.append("--baseline")
        if args.claim:
            flags.extend(["--claim", args.claim])
        return _run(["-m", "research_verifier.runtime.smoke", *flags])
    if args.command == "setup-models":
        steps = [
            ["-m", "paper_search.model_setup", "--model-path", str(RETRIEVAL_MODEL_DIR)],
            ["-m", "research_verifier.runtime.setup_verifier"],
            ["-m", "research_verifier.runtime.setup_evidence"],
        ]
        if args.language_model:
            steps.append(["-m", "research_verifier.runtime.setup_language_model"])
        for step in steps:
            result = _run(step)
            if result:
                return result
        return 0
    return _run([str(PROJECT_ROOT / "scripts" / "check_integrity.py")])


if __name__ == "__main__":
    raise SystemExit(main())
