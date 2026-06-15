#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START="${START:-2000}"
END="${END:-2999}"
CAMERAS="${CAMERAS:-cam0}"
GPU_ID="${GPU_ID:-0}"
INPUT_DIR="${INPUT_DIR:-/root/epfs/new_route_stage1_skymask/sam2_input_${START}_${END}}"
INPUT_MASK_DIR="${INPUT_MASK_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_trt_candidate_rle50}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_trt_rle50_coverage_completed}"
SKY_MASK_DIR="${SKY_MASK_DIR:-/root/epfs/new_route_data/sky_masks_color}"
REPORT="${REPORT:-${OUTPUT_DIR}/coverage_completion_report.json}"
PYTHON="${PYTHON:-/root/epfs/conda_envs/vlm_seg/bin/python}"

TARGET_COVERAGE="${TARGET_COVERAGE:-0.90}"
MAX_PROMPT_POINTS="${MAX_PROMPT_POINTS:-24}"
MIN_UNCOVERED_COMPONENT_AREA="${MIN_UNCOVERED_COMPONENT_AREA:-3000}"
UNCOVERED_AREA_PER_POINT="${UNCOVERED_AREA_PER_POINT:-60000}"
MIN_NEW_AREA="${MIN_NEW_AREA:-800}"
MAX_OVERLAP_RATIO="${MAX_OVERLAP_RATIO:-0.95}"
MIN_MEDIAN_LUMA_FOR_COMPLETION="${MIN_MEDIAN_LUMA_FOR_COMPLETION:-25}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${OUTPUT_DIR}"

image_globs=()
IFS=',' read -r -a camera_list <<< "${CAMERAS}"
for camera in "${camera_list[@]}"; do
  [[ -n "${camera}" ]] || continue
  image_globs+=("${INPUT_DIR}/${camera}_*.png")
done
if [[ "${#image_globs[@]}" -eq 0 ]]; then
  echo "no cameras selected" >&2
  exit 2
fi

tmp_list="${OUTPUT_DIR}/coverage_completion_images.txt"
: > "${tmp_list}"
for pattern in "${image_globs[@]}"; do
  find "$(dirname "${pattern}")" -maxdepth 1 \( -type f -o -type l \) -name "$(basename "${pattern}")"
done | sort -u > "${tmp_list}"

if [[ ! -s "${tmp_list}" ]]; then
  echo "no images found under ${INPUT_DIR} for CAMERAS=${CAMERAS}" >&2
  exit 2
fi

args=(
  "${SCRIPT_DIR}/complete_sam2_masks_by_coverage.py"
  --images "@${tmp_list}"
  --input-mask-dir "${INPUT_MASK_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --sky-mask-dir "${SKY_MASK_DIR}"
  --target-coverage "${TARGET_COVERAGE}"
  --max-prompt-points "${MAX_PROMPT_POINTS}"
  --min-uncovered-component-area "${MIN_UNCOVERED_COMPONENT_AREA}"
  --uncovered-area-per-point "${UNCOVERED_AREA_PER_POINT}"
  --min-new-area "${MIN_NEW_AREA}"
  --max-overlap-ratio "${MAX_OVERLAP_RATIO}"
  --min-median-luma-for-completion "${MIN_MEDIAN_LUMA_FOR_COMPLETION}"
  --report "${REPORT}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

echo "input_masks=${INPUT_MASK_DIR}"
echo "output=${OUTPUT_DIR}"
echo "images=$(wc -l < "${tmp_list}")"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${args[@]}"
