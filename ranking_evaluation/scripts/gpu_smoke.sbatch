#!/bin/bash
#SBATCH --job-name=ranking-gpu-smoke
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:10:00
#SBATCH --output=outputs/ranking_evaluation/gpu-smoke-%j.log

set -euo pipefail
: "${RANKING_RUNTIME:?Set RANKING_RUNTIME to your SSD runtime directory}"
: "${RANKING_REPO:?Set RANKING_REPO to the repository root}"
export TMPDIR="$RANKING_RUNTIME/.tmp/$SLURM_JOB_ID"
export PIP_CACHE_DIR="$RANKING_RUNTIME/.tmp/pip"
export CONDA_PKGS_DIRS="$RANKING_RUNTIME/.tmp/conda"
export HF_HOME="$RANKING_RUNTIME/.tmp/hf"
export TRITON_CACHE_DIR="$RANKING_RUNTIME/.tmp/triton"
export XDG_CACHE_HOME="$RANKING_RUNTIME/.tmp/cache"
mkdir -p "$TMPDIR"
export PYTHONPATH="$RANKING_REPO/live_search/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$RANKING_REPO"
nvidia-smi
"$RANKING_RUNTIME/env/bin/python" -m ranking_evaluation.examples.verify_gpu_reranker \
  --model-path "$RANKING_RUNTIME/model" \
  --output "outputs/ranking_evaluation/gpu-smoke-$SLURM_JOB_ID.json"
