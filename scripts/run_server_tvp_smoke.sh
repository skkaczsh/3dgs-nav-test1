#!/usr/bin/env bash
set -euo pipefail

TVP_REPO="${TVP_REPO:-/root/epfs/model_side_tracks/tvp/Thinking-with-Visual-Primitives-pytorch}"
MANIFEST="${MANIFEST:-/root/epfs/model_side_tracks/tvp/tvp_candidate_manifest_10.json}"
MODEL_PATH="${MODEL_PATH:-yunfengwang/TVP-OPD-Qwen2VL-2B}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/root/epfs/model_side_tracks/tvp/tvp_raw_smoke.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
TVP_DOWNLOAD_DIR="${TVP_DOWNLOAD_DIR:-/root/epfs/model_side_tracks/tvp/hf_cache}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$TVP_DOWNLOAD_DIR"

ARGS=(
  "$SCRIPT_DIR/run_tvp_manifest_inference.py"
  --tvp-repo "$TVP_REPO"
  --manifest "$MANIFEST"
  --model-path "$MODEL_PATH"
  --output-jsonl "$OUTPUT_JSONL"
  --download-dir "$TVP_DOWNLOAD_DIR"
  --max-samples "$MAX_SAMPLES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
)

if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  ARGS+=(--load-in-4bit)
fi

"$PYTHON_BIN" "${ARGS[@]}"
echo "$OUTPUT_JSONL"
