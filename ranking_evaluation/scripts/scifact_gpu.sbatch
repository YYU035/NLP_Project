#!/bin/bash
#SBATCH --job-name=ranking-scifact
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/ranking_evaluation/scifact-%j.log

set -euo pipefail
: "${RANKING_RUNTIME:?Set RANKING_RUNTIME to your SSD runtime directory}"
: "${RANKING_REPO:?Set RANKING_REPO to the repository root}"
: "${NAILA_SOURCE:?Set NAILA_SOURCE to the pinned Person 2 snapshot}"
export TMPDIR="$RANKING_RUNTIME/.tmp/$SLURM_JOB_ID"
export PIP_CACHE_DIR="$RANKING_RUNTIME/.tmp/pip"
export HF_HOME="$RANKING_RUNTIME/.tmp/hf"
export TRITON_CACHE_DIR="$RANKING_RUNTIME/.tmp/triton"
export XDG_CACHE_HOME="$RANKING_RUNTIME/.tmp/cache"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$RANKING_REPO/live_search/src"
mkdir -p "$TMPDIR"
cd "$RANKING_REPO"
RANKING_PYTHON="$RANKING_RUNTIME/env/bin/python"
RANKING_OUTPUT="$RANKING_RUNTIME/experiments/scifact-dev-$SLURM_JOB_ID"
mkdir -p "$RANKING_OUTPUT"
ln -s "$RANKING_OUTPUT" "outputs/ranking_evaluation/scifact-dev-$SLURM_JOB_ID"
# Freeze protocol and sources before inference. Dev is already used elsewhere
# in the project; this run is development evidence, not untouched final testing.
"$RANKING_PYTHON" - "$RANKING_REPO" "$RANKING_OUTPUT" "$NAILA_SOURCE" <<'PY'
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
repo,out,upstream=map(Path,sys.argv[1:])
source=out/'source'
for folder in ('ranking_evaluation','live_search/src/paper_search'):
    for path in (repo/folder).rglob('*.py'):
        target=source/path.relative_to(repo)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
shutil.copytree(repo/'ranking_evaluation/experiments',source/'ranking_evaluation/experiments')
(out/'protocol.json').write_text(json.dumps({
    'status':'declared before inference','split':'dev','purpose':'full development comparison; not untouched final evaluation',
    'claims':'all split claims retained; no-gold excluded from relevance means with counts',
    'corpus':'complete SciFact corpus; no gold candidate filtering',
    'branch_depth':100,'rrf_k':60,'rerank_depths':[10,20,50],'repeats':1,
    'model_selection':'none; all declared configurations reported',
    'latency':'single run includes cold inference; not stable throughput or end-to-end latency',
    'upstream_source':json.loads((upstream/'source_manifest.json').read_text()),
    'source_sha256':{p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()},
    'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),
},indent=2)+'\n')
PY
cd "$RANKING_OUTPUT/source"
export PYTHONPATH="$RANKING_OUTPUT/source/live_search/src"
nvidia-smi
"$RANKING_PYTHON" -m pip freeze > "$RANKING_OUTPUT/environment.txt"
"$RANKING_PYTHON" -m ranking_evaluation.examples.verify_gpu_reranker \
    --model-path "$RANKING_RUNTIME/model" --output "$RANKING_OUTPUT/gpu-smoke.json"
SBERT_REVISION=$("$RANKING_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$RANKING_RUNTIME/sbert-model/download_manifest.json")
"$RANKING_PYTHON" -m ranking_evaluation.export_naila \
    --retrieval-root "$NAILA_SOURCE" \
    --corpus "$RANKING_RUNTIME/scifact/corpus.jsonl" \
    --claims "$RANKING_RUNTIME/scifact/claims_dev.jsonl" --split dev \
    --model-path "$RANKING_RUNTIME/sbert-model" \
    --model-id sentence-transformers/all-MiniLM-L6-v2 --model-revision "$SBERT_REVISION" \
    --device cuda --k 100 --output "$RANKING_OUTPUT/rankings.json"
"$RANKING_PYTHON" -m ranking_evaluation.scifact \
    --corpus "$RANKING_RUNTIME/scifact/corpus.jsonl" \
    --claims "$RANKING_RUNTIME/scifact/claims_dev.jsonl" --split dev \
    --rankings "$RANKING_OUTPUT/rankings.json" --output "$RANKING_OUTPUT/bundle.json"
"$RANKING_PYTHON" -m ranking_evaluation.experiment_suite \
    --input "$RANKING_OUTPUT/bundle.json" \
    --config ranking_evaluation/experiments/rerank-pilot.json \
    --output-dir "$RANKING_OUTPUT/comparison" \
    --model-path "$RANKING_RUNTIME/model" \
    --model-id cross-encoder/ms-marco-MiniLM-L6-v2 \
    --model-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a --device cuda
