#!/usr/bin/env bash
set -euo pipefail

# Production-shaped SAM2 TensorRT mask generation.
#
# This writes production-compatible SAM mask artifacts into a candidate
# directory. It does not overwrite the validated Python SAM2 mask directory. If
# a baseline directory is provided, it also runs a promotion gate and reports
# pass/fail.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START="${START:-2000}"
END="${END:-2999}"
GPU_ID="${GPU_ID:-0}"
INPUT_DIR="${INPUT_DIR:-/root/epfs/new_route_stage1_skymask/sam2_input_${START}_${END}}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_trt_candidate}"
REPORT_DIR="${REPORT_DIR:-/root/epfs/sam2_tensorrt/reports/production_${START}_${END}}"
BASELINE_DIR="${BASELINE_DIR:-}"
RUNNER="${RUNNER:-/root/epfs/sam2_tensorrt/bin/sam2_trt_amg_runner}"
RUNNER_SRC="${RUNNER_SRC:-/root/epfs/new_route_tools/sam2_trt_amg_runner.cpp}"
IMAGE_GLOB="${IMAGE_GLOB:-${INPUT_DIR}/*.png}"
OUTPUT_MODE="${OUTPUT_MODE:-uncompressed_rle}"
OVERWRITE="${OVERWRITE:-0}"
SEMANTIC_EVAL_RUN_EVAL="${SEMANTIC_EVAL_RUN_EVAL:-/root/epfs/manifold_3dgs_project/semantic_eval/run_eval.py}"
PATCH_SEMANTIC_EVAL_RLE="${PATCH_SEMANTIC_EVAL_RLE:-1}"

POINTS_PER_SIDE="${POINTS_PER_SIDE:-32}"
POINTS_PER_BATCH="${POINTS_PER_BATCH:-64}"
PRED_IOU_THRESH="${PRED_IOU_THRESH:-0.7}"
STABILITY_SCORE_THRESH="${STABILITY_SCORE_THRESH:-0.92}"
BOX_NMS_THRESH="${BOX_NMS_THRESH:-0.7}"
CROP_NMS_THRESH="${CROP_NMS_THRESH:-0.65}"
MIN_MASK_AREA="${MIN_MASK_AREA:-500}"
CROP_N_LAYERS="${CROP_N_LAYERS:-1}"

MIN_OK_IMAGES="${MIN_OK_IMAGES:-10}"
MIN_MEAN_MATCHED_IOU="${MIN_MEAN_MATCHED_IOU:-0.93}"
MAX_ABS_MEAN_COVERAGE_DELTA="${MAX_ABS_MEAN_COVERAGE_DELTA:-0.06}"
MAX_ABS_ROW_COVERAGE_DELTA="${MAX_ABS_ROW_COVERAGE_DELTA:-0.25}"
MAX_MEAN_UNMATCHED_BASELINE="${MAX_MEAN_UNMATCHED_BASELINE:-4.0}"
MAX_MEAN_UNMATCHED_CANDIDATE="${MAX_MEAN_UNMATCHED_CANDIDATE:-8.0}"

mkdir -p "${OUTPUT_DIR}" "${REPORT_DIR}"

if [[ ! -x "${RUNNER}" || "${BUILD_RUNNER:-0}" == "1" ]]; then
  SRC="${RUNNER_SRC}" OUT="${RUNNER}" bash "${SCRIPT_DIR}/build_sam2_tensorrt_runner.sh"
fi

if [[ "${OUTPUT_MODE}" == "uncompressed_rle" && "${PATCH_SEMANTIC_EVAL_RLE}" == "1" && -f "${SEMANTIC_EVAL_RUN_EVAL}" ]]; then
  echo "[0/4] patching semantic_eval RLE mask loader"
  python3 "${SCRIPT_DIR}/patch_semantic_eval_rle_masks.py" --run-eval "${SEMANTIC_EVAL_RUN_EVAL}"
fi

runner_args=(
  --images "${IMAGE_GLOB}"
  --output-dir "${OUTPUT_DIR}"
  --points-per-side "${POINTS_PER_SIDE}"
  --points-per-batch "${POINTS_PER_BATCH}"
  --crop-n-layers "${CROP_N_LAYERS}"
  --output-mode "${OUTPUT_MODE}"
  --pred-iou-thresh "${PRED_IOU_THRESH}"
  --stability-score-thresh "${STABILITY_SCORE_THRESH}"
  --box-nms-thresh "${BOX_NMS_THRESH}"
  --crop-nms-thresh "${CROP_NMS_THRESH}"
  --min-mask-area "${MIN_MASK_AREA}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  runner_args+=(--overwrite)
fi

echo "[1/4] running SAM2 TensorRT production candidate"
echo "input=${IMAGE_GLOB}"
echo "output=${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${RUNNER}" "${runner_args[@]}" \
  >"${REPORT_DIR}/sam2_trt_runner.stdout.jsonl" \
  2>"${REPORT_DIR}/sam2_trt_runner.stderr.log"

echo "[2/4] writing production manifest"
python3 - "${IMAGE_GLOB}" "${OUTPUT_DIR}" "${REPORT_DIR}/sam2_trt_production_manifest.json" <<'PY'
import glob
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[2])
manifest = Path(sys.argv[3])
images = [Path(p) for p in sorted(glob.glob(sys.argv[1]))]
items = []
missing = []
for image in images:
    image_id = image.stem
    if (output_dir / f"{image_id}_sam_masks.json").exists():
        items.append({"image_id": image_id, "image_path": str(image)})
    else:
        missing.append(image_id)
report = {
    "image_glob": sys.argv[1],
    "output_dir": str(output_dir),
    "images": len(images),
    "candidate_masks": len(items),
    "missing": len(missing),
    "missing_samples": missing[:100],
    "items": items,
}
manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: report[k] for k in ["images", "candidate_masks", "missing"]}, ensure_ascii=False, indent=2))
PY

if [[ -n "${BASELINE_DIR}" ]]; then
  echo "[3/4] comparing candidate against Python SAM2 baseline"
  /root/epfs/conda_envs/vlm_seg/bin/python "${SCRIPT_DIR}/compare_sam_mask_dirs.py" \
    --baseline-dir "${BASELINE_DIR}" \
    --candidate-dir "${OUTPUT_DIR}" \
    --manifest "${REPORT_DIR}/sam2_trt_production_manifest.json" \
    --json-output "${REPORT_DIR}/sam2_trt_production_compare.json" \
    --csv-output "${REPORT_DIR}/sam2_trt_production_compare.csv"

  echo "[4/4] running promotion gate"
  /root/epfs/conda_envs/vlm_seg/bin/python "${SCRIPT_DIR}/gate_sam2_trt_promotion.py" \
    --compare-json "${REPORT_DIR}/sam2_trt_production_compare.json" \
    --output "${REPORT_DIR}/sam2_trt_promotion_gate.json" \
    --min-ok-images "${MIN_OK_IMAGES}" \
    --min-mean-matched-iou "${MIN_MEAN_MATCHED_IOU}" \
    --max-abs-mean-coverage-delta "${MAX_ABS_MEAN_COVERAGE_DELTA}" \
    --max-abs-row-coverage-delta "${MAX_ABS_ROW_COVERAGE_DELTA}" \
    --max-mean-unmatched-baseline "${MAX_MEAN_UNMATCHED_BASELINE}" \
    --max-mean-unmatched-candidate "${MAX_MEAN_UNMATCHED_CANDIDATE}"
else
  echo "[3/4] baseline comparison skipped: BASELINE_DIR is empty"
  echo "[4/4] promotion gate skipped"
fi

echo "candidate_dir=${OUTPUT_DIR}"
echo "report_dir=${REPORT_DIR}"
