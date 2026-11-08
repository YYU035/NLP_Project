#!/bin/bash
#SBATCH --job-name=person3-finish
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=outputs/ranking_evaluation/semantic-%j.log
set -euo pipefail
: "${RANKING_RUNTIME:?Set SSD runtime root}"
: "${RANKING_REPO:?Set repository root}"
: "${NAILA_SOURCE:?Set pinned Person 2 source}"
export TMPDIR="$RANKING_RUNTIME/.tmp/$SLURM_JOB_ID"
export HF_HOME="$RANKING_RUNTIME/.tmp/hf" XDG_CACHE_HOME="$RANKING_RUNTIME/.tmp/cache"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
RUN="$RANKING_RUNTIME/experiments/semantic-evidence-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$RUN/source"
cd "$RANKING_REPO"
ln -s "$RUN" "outputs/ranking_evaluation/semantic-evidence-$SLURM_JOB_ID"
"$RANKING_RUNTIME/env/bin/python" - "$RANKING_REPO" "$RUN/source" <<'PY'
from pathlib import Path
import sys,shutil
root,dest=map(Path,sys.argv[1:])
for folder in ('ranking_evaluation','live_search/src/paper_search'):
 for p in (root/folder).rglob('*.py'):
  target=dest/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
PY
cd "$RUN/source"
export PYTHONPATH="$RUN/source/live_search/src"
"$RANKING_RUNTIME/env/bin/python" -m ranking_evaluation.semantic_experiment \
  --runtime "$RANKING_RUNTIME" \
  --comparison "$RANKING_RUNTIME/experiments/scifact-dev-276357/comparison" \
  --retrieval-root "$NAILA_SOURCE" --output-dir "$RUN/results"
