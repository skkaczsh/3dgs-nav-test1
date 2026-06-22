#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_GLOB="${IMAGE_GLOB:-/root/epfs/new_route_stage1_skymask/sam2_input_2000_2999/cam0_00200[0-9].png}"
BASELINE_DIR="${BASELINE_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_combined}"
CANDIDATE_DIR="${CANDIDATE_DIR:-/root/epfs/sam2_tensorrt/sam_masks_candidate_benchmark}"
REPORT_DIR="${REPORT_DIR:-/root/epfs/sam2_tensorrt/reports}"
RUNNER="${RUNNER:-${REPO_ROOT}/build/sam2_tensorrt/bin/sam2_trt_amg_runner}"
RUNNER_SRC="${RUNNER_SRC:-${REPO_ROOT}/tools/sam2_trt_amg_runner.cpp}"
OUTPUT_MODE="${OUTPUT_MODE:-uncompressed_rle}"
PRED_IOU_THRESH="${PRED_IOU_THRESH:-0.7}"
STABILITY_SCORE_THRESH="${STABILITY_SCORE_THRESH:-0.92}"
BOX_NMS_THRESH="${BOX_NMS_THRESH:-0.7}"
CROP_NMS_THRESH="${CROP_NMS_THRESH:-0.65}"
MIN_MASK_AREA="${MIN_MASK_AREA:-500}"
mkdir -p "${CANDIDATE_DIR}" "${REPORT_DIR}"

if [[ ! -x "${RUNNER}" ]]; then
  SRC="${RUNNER_SRC}" OUT="${RUNNER}" bash "${SCRIPT_DIR}/build_sam2_tensorrt_runner.sh"
fi

echo "[1/3] running SAM2 TensorRT AMG benchmark candidate"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${RUNNER}" \
  --images "${IMAGE_GLOB}" \
  --output-dir "${CANDIDATE_DIR}" \
  --crop-n-layers 1 \
  --output-mode "${OUTPUT_MODE}" \
  --pred-iou-thresh "${PRED_IOU_THRESH}" \
  --stability-score-thresh "${STABILITY_SCORE_THRESH}" \
  --box-nms-thresh "${BOX_NMS_THRESH}" \
  --crop-nms-thresh "${CROP_NMS_THRESH}" \
  --min-mask-area "${MIN_MASK_AREA}" \
  --overwrite

MANIFEST="${REPORT_DIR}/sam2_trt_benchmark_manifest.json"
echo "[2/3] writing candidate manifest: ${MANIFEST}"
python3 - "${CANDIDATE_DIR}" "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

candidate_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
items = [
    {"image_id": p.name.removesuffix("_sam_masks.json")}
    for p in sorted(candidate_dir.glob("*_sam_masks.json"))
]
manifest.write_text(json.dumps({"items": items}, indent=2), encoding="utf-8")
print(f"candidate_images={len(items)}")
PY

echo "[3/3] comparing against Python SAM2 baseline"
/root/epfs/conda_envs/vlm_seg/bin/python "${SCRIPT_DIR}/compare_sam_mask_dirs.py" \
  --baseline-dir "${BASELINE_DIR}" \
  --candidate-dir "${CANDIDATE_DIR}" \
  --manifest "${MANIFEST}" \
  --json-output "${REPORT_DIR}/sam2_trt_benchmark_compare.json" \
  --csv-output "${REPORT_DIR}/sam2_trt_benchmark_compare.csv"
