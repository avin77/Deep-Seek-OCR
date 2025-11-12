#!/usr/bin/env bash

set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
FASTAPI_PORT="${FASTAPI_PORT:-9000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-OCR}"

run_vllm() {
    if ! command -v vllm &>/dev/null; then
        echo "vLLM is not installed. Install with 'pip install vllm'." >&2
        exit 1
    fi

    echo "Starting DeepSeek-OCR via vLLM on port ${VLLM_PORT}..."
    vllm serve "${MODEL_PATH}" \
        --port "${VLLM_PORT}" \
        --api-key local-placeholder \
        --enable-auto-tool-choice false \
        --enforce-eager \
        --trust-remote-code \
        --max-model-len 32768
}

run_api() {
    echo "Launching FastAPI service on port ${FASTAPI_PORT}..."
    uvicorn app.api.main:app \
        --host 0.0.0.0 \
        --port "${FASTAPI_PORT}" \
        --reload
}

run_ui() {
    echo "Launching Streamlit reviewer on port ${STREAMLIT_PORT}..."
    streamlit run ui/app.py --server.port "${STREAMLIT_PORT}"
}

usage() {
    cat <<EOF
Usage: scripts/run_local.sh [vllm|api|ui|all]

Environment variables:
  MODEL_PATH     Hugging Face model name or local checkpoint path (default: ${MODEL_PATH})
  VLLM_PORT      Port for the vLLM OpenAI-compatible server (default: ${VLLM_PORT})
  FASTAPI_PORT   Port for uvicorn (default: ${FASTAPI_PORT})
  STREAMLIT_PORT Port for Streamlit (default: ${STREAMLIT_PORT})

Examples:
  MODEL_PATH=/models/DeepSeek-OCR scripts/run_local.sh vllm
  FASTAPI_PORT=9100 scripts/run_local.sh api
  scripts/run_local.sh all        # start vLLM, API, and UI sequentially
EOF
}

case "${1:-help}" in
    vllm) run_vllm ;;
    api) run_api ;;
    ui) run_ui ;;
    all)
        run_vllm &
        VLLM_PID=$!
        sleep 10
        run_api &
        API_PID=$!
        sleep 5
        run_ui &
        UI_PID=$!
        trap 'kill ${UI_PID:-0} ${API_PID:-0} ${VLLM_PID:-0}' INT TERM
        wait
        ;;
    help|--help|-h) usage ;;
    *)
        usage
        exit 1
        ;;
esac
