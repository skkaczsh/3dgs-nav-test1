#!/usr/bin/env bash
set -euo pipefail

# Build SAM2 evidence only for already selected shared contact views.
#
# This script never changes Superpoint ownership or semantic labels. Its output
# is a conservative edge multiplier: repeated compact-mask separation may lower
# a contact affinity; absent or one-view evidence stays neutral.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-${ROOT}/build/sam2_tensorrt/bin/sam2_trt_amg_runner}"
GPU_ID="${GPU_ID:-0}"
# Comma-separated physical GPU ids.  With one id this is exactly the historic
# runner behavior; with several ids unique camera views are deterministically
# sharded and all workers write distinct mask filenames into MASK_DIR.
GPU_IDS="${GPU_IDS:-$GPU_ID}"
POINTS_PER_SIDE="${POINTS_PER_SIDE:-32}"
# The deployed TensorRT decoder engine has a fixed prompt batch dimension of
# 64.  Changing this requires rebuilding that engine; do not expose it as a
# fake runtime tuning knob.
POINTS_PER_BATCH="${POINTS_PER_BATCH:-64}"
MIN_MASK_AREA="${MIN_MASK_AREA:-500}"

usage() {
  cat <<'EOF'
Usage:
  run_superpoint_sam2_edge_evidence.sh \
    --evidence-jsonl EVIDENCE.jsonl \
    --contact-edges CONTACT.jsonl \
    --output-dir OUTPUT_DIR

Legacy two-ledger form:
  run_superpoint_sam2_edge_evidence.sh \
    --edge-evidence EDGE_ONLY.jsonl \
    --direct-evidence DIRECT.jsonl \
    --contact-edges CONTACT.jsonl \
    --output-dir OUTPUT_DIR
EOF
}

EDGE_EVIDENCE=""
DIRECT_EVIDENCE=""
EVIDENCE_JSONL=""
CONTACT_EDGES=""
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --edge-evidence) EDGE_EVIDENCE="$2"; shift 2 ;;
    --direct-evidence) DIRECT_EVIDENCE="$2"; shift 2 ;;
    --evidence-jsonl) EVIDENCE_JSONL="$2"; shift 2 ;;
    --contact-edges) CONTACT_EDGES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$EVIDENCE_JSONL" ]]; then
  [[ -z "$EDGE_EVIDENCE" && -z "$DIRECT_EVIDENCE" ]] || {
    echo "--evidence-jsonl is mutually exclusive with --edge-evidence/--direct-evidence" >&2; exit 2;
  }
  [[ -f "$EVIDENCE_JSONL" ]] || { echo "Missing required input: $EVIDENCE_JSONL" >&2; exit 2; }
else
  for path in "$EDGE_EVIDENCE" "$DIRECT_EVIDENCE"; do
    [[ -n "$path" && -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
  done
fi
[[ -n "$CONTACT_EDGES" && -f "$CONTACT_EDGES" ]] || { echo "Missing required input: $CONTACT_EDGES" >&2; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { echo "--output-dir is required" >&2; exit 2; }
[[ -x "$RUNNER" ]] || { echo "TensorRT SAM2 runner is not executable: $RUNNER" >&2; exit 2; }

INPUT_DIR="${OUTPUT_DIR}/sam2_inputs"
MASK_DIR="${OUTPUT_DIR}/sam_masks"
COMBINED_EVIDENCE="${OUTPUT_DIR}/evidence_with_shared_neighbors.jsonl"
mkdir -p "$OUTPUT_DIR" "$MASK_DIR"

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/make_sam2_input_links.py" \
  --views-jsonl "${EVIDENCE_JSONL:-$EDGE_EVIDENCE}" \
  --output-dir "$INPUT_DIR" \
  --report "${OUTPUT_DIR}/sam2_input_report.json"

shopt -s nullglob
image_count=$(find "$INPUT_DIR" -maxdepth 1 -type l \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
(( image_count > 0 )) || { echo "No images linked from edge evidence" >&2; exit 2; }

: > "${OUTPUT_DIR}/sam2_runner.stdout.jsonl"
: > "${OUTPUT_DIR}/sam2_runner.stderr.log"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
(( ${#GPU_LIST[@]} > 0 )) || { echo "GPU_IDS is empty" >&2; exit 2; }
SHARD_ROOT="${OUTPUT_DIR}/sam2_input_shards"
rm -rf "$SHARD_ROOT"
mkdir -p "$SHARD_ROOT"
mapfile -t INPUT_IMAGES < <(find "$INPUT_DIR" -maxdepth 1 -type l \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print | sort)
for index in "${!INPUT_IMAGES[@]}"; do
  shard=$(( index % ${#GPU_LIST[@]} ))
  shard_dir="$SHARD_ROOT/$shard"
  mkdir -p "$shard_dir"
  ln -s "${INPUT_IMAGES[$index]}" "$shard_dir/$(basename "${INPUT_IMAGES[$index]}")"
done

run_sam2_shard() {
  local shard="$1"
  local gpu="$2"
  local shard_dir="$SHARD_ROOT/$shard"
  local stdout_path="${OUTPUT_DIR}/sam2_runner_gpu${gpu}.stdout.jsonl"
  local stderr_path="${OUTPUT_DIR}/sam2_runner_gpu${gpu}.stderr.log"
  : > "$stdout_path"
  : > "$stderr_path"
  for extension in jpg jpeg png; do
    local pattern="${shard_dir}/*.${extension}"
    compgen -G "$pattern" >/dev/null || continue
    CUDA_VISIBLE_DEVICES="$gpu" "$RUNNER" \
      --images "$pattern" \
      --output-dir "$MASK_DIR" \
      --points-per-side "$POINTS_PER_SIDE" \
      --points-per-batch "$POINTS_PER_BATCH" \
      --crop-n-layers 1 \
      --output-mode compressed_rle \
      --pred-iou-thresh 0.7 \
      --stability-score-thresh 0.92 \
      --box-nms-thresh 0.7 \
      --crop-nms-thresh 0.65 \
      --min-mask-area "$MIN_MASK_AREA" \
      --skip-visuals \
      >> "$stdout_path" \
      2>> "$stderr_path"
  done
}

pids=()
for shard in "${!GPU_LIST[@]}"; do
  run_sam2_shard "$shard" "${GPU_LIST[$shard]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
cat "${OUTPUT_DIR}"/sam2_runner_gpu*.stdout.jsonl > "${OUTPUT_DIR}/sam2_runner.stdout.jsonl"
cat "${OUTPUT_DIR}"/sam2_runner_gpu*.stderr.log > "${OUTPUT_DIR}/sam2_runner.stderr.log"

if [[ -n "$EVIDENCE_JSONL" ]]; then
  cp "$EVIDENCE_JSONL" "$COMBINED_EVIDENCE"
else
  cat "$DIRECT_EVIDENCE" "$EDGE_EVIDENCE" > "$COMBINED_EVIDENCE"
fi
PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/build_superpoint_sam2_comask_edges.py" \
  --evidence-jsonl "$COMBINED_EVIDENCE" \
  --contact-edges "$CONTACT_EDGES" \
  --sam-mask-dir "$MASK_DIR" \
  --output-jsonl "${OUTPUT_DIR}/sam2_comask_edges.jsonl" \
  --report "${OUTPUT_DIR}/sam2_comask_report.json"

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/make_superpoint_sam2_edge_review.py" \
  --sam2-edges "${OUTPUT_DIR}/sam2_comask_edges.jsonl" \
  --evidence-jsonl "$COMBINED_EVIDENCE" \
  --sam-mask-dir "$MASK_DIR" \
  --output-dir "${OUTPUT_DIR}/sam2_edge_review"

echo "output_dir=${OUTPUT_DIR}"
