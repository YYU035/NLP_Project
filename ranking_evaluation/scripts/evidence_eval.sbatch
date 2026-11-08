#!/bin/bash
#SBATCH --job-name=evidence-evaluation
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=outputs/ranking_evaluation/evidence-%j.log

set -euo pipefail
: "${RANKING_RUNTIME:?Set SSD runtime root}"
: "${RANKING_REPO:?Set repository root}"
: "${RANKING_COMPARISON:?Set directory containing complete ranking reports}"
export TMPDIR="$RANKING_RUNTIME/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
EVIDENCE_RUN="$RANKING_RUNTIME/experiments/evidence-dev-$SLURM_JOB_ID"
mkdir -p "$EVIDENCE_RUN/source"
cd "$RANKING_REPO"
ln -s "$EVIDENCE_RUN" "outputs/ranking_evaluation/evidence-dev-$SLURM_JOB_ID"
"$RANKING_RUNTIME/env/bin/python" - "$RANKING_REPO" "$EVIDENCE_RUN/source" <<'PY'
from pathlib import Path
import sys,shutil
root,dest=map(Path,sys.argv[1:])
for folder in ('ranking_evaluation','live_search/src/paper_search'):
    for p in (root/folder).rglob('*.py'):
        target=dest/p.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
PY
cd "$EVIDENCE_RUN/source"
export PYTHONPATH="$EVIDENCE_RUN/source/live_search/src"
"$RANKING_RUNTIME/env/bin/python" -m ranking_evaluation.evaluate_evidence \
  --corpus "$RANKING_RUNTIME/scifact/corpus.jsonl" \
  --claims "$RANKING_RUNTIME/scifact/claims_dev.jsonl" --split dev \
  --ranking-reports \
    "$RANKING_COMPARISON/repeat-1-rrf-60.json" \
    "$RANKING_COMPARISON/repeat-1-rrf-60-rerank-10.json" \
    "$RANKING_COMPARISON/repeat-1-rrf-60-rerank-20.json" \
    "$RANKING_COMPARISON/repeat-1-rrf-60-rerank-50.json" \
  --paper-depths 5 10 --output-dir "$EVIDENCE_RUN/results"
