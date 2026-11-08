#!/bin/bash
set -euo pipefail
: "${FRONTEND_ROOT:?Set Kireeti checkout root}"
: "${FRONTEND_PYTHON:?Set frontend environment Python}"
: "${PERSON3_EVIDENCE_CONFIG:?Set frozen selector JSON}"
: "${PERSON3_EVIDENCE_MODEL_PATH:?Set local pinned model directory}"
: "${PERSON3_VERIFIER_TOKENIZER_PATH:?Set Person 4 pinned local tokenizer directory}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false HF_HUB_DISABLE_TELEMETRY=1
exec "$FRONTEND_PYTHON" -m streamlit run "$FRONTEND_ROOT/explanation_frontend/app.py" \
  --server.address 127.0.0.1 --server.port "${FRONTEND_PORT:-18503}"
