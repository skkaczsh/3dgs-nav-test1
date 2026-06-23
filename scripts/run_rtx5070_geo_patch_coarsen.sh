#!/usr/bin/env bash
set -euo pipefail

# Run the coarse supernode diagnostic stage on scan-rtx5070.
#
# This consumes an existing 0.03m C++ region-grow output and coarsens it into
# a smaller candidate graph.  It is a diagnostic object-candidate layer, not a
# semantic classifier.

REMOTE_HOST="${REMOTE_HOST:-zsh@skkac.top}"
REMOTE_PORT="${REMOTE_PORT:-6010}"
REMOTE_KEY="${REMOTE_KEY:-/Users/skkac/.ssh/id_ed25519}"
REMOTE_REPO="${REMOTE_REPO:-/home/zsh/Work/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
REMOTE_VENV="${REMOTE_VENV:-/home/zsh/Work/SCAN/.venvs/scan-semantic}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python}"
TMUX_SESSION="${TMUX_SESSION:-scan_patch_coarsen_20k}"

RUN="${RUN:-0}"
REGION_RUN="${REGION_RUN:-${REMOTE_WORK}/geo_patch_5070_full_0000_6180_energy_v3_voxel003_20260624_0018}"
OUT_NAME="${OUT_NAME:-geo_patch_5070_full_0000_6180_coarse20k_voxel003_20260624_0045}"
OUTPUT_DIR="${OUTPUT_DIR:-${REMOTE_WORK}/${OUT_NAME}}"

TARGET_PATCHES="${TARGET_PATCHES:-20000}"
NOISE_PATCH_VOXELS="${NOISE_PATCH_VOXELS:-3}"
PRECOLLAPSE_MODE="${PRECOLLAPSE_MODE:-connected-grid}"
PRECOLLAPSE_GRID_SIZE="${PRECOLLAPSE_GRID_SIZE:-0.50}"
PRECOLLAPSE_MIN_COMPONENT_VOXELS="${PRECOLLAPSE_MIN_COMPONENT_VOXELS:-24}"
NEIGHBORS_PER_PATCH="${NEIGHBORS_PER_PATCH:-18}"
MAX_CENTROID_DISTANCE="${MAX_CENTROID_DISTANCE:-2.5}"
MAX_BBOX_GAP="${MAX_BBOX_GAP:-0.45}"
MIN_MERGE_SCORE="${MIN_MERGE_SCORE:-0.56}"
MAX_COLOR_DISTANCE="${MAX_COLOR_DISTANCE:-170}"
HARD_COLOR_DISTANCE="${HARD_COLOR_DISTANCE:-230}"
MAX_COMPONENT_VOXELS="${MAX_COMPONENT_VOXELS:-180000}"
PREVIEW_STRIDE="${PREVIEW_STRIDE:-5}"
OVERLAP_VOXEL_SIZE="${OVERLAP_VOXEL_SIZE:-0.20}"
OVERLAP_MIN_RATIO="${OVERLAP_MIN_RATIO:-0.35}"
OVERLAP_MIN_SCORE="${OVERLAP_MIN_SCORE:-0.62}"

SSH=(ssh -i "${REMOTE_KEY}" -p "${REMOTE_PORT}" "${REMOTE_HOST}")
RSYNC_SSH="ssh -i ${REMOTE_KEY} -p ${REMOTE_PORT}"

echo "remote=${REMOTE_HOST}:${REMOTE_PORT}"
echo "region_run=${REGION_RUN}"
echo "output_dir=${OUTPUT_DIR}"
if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

rsync -az -e "${RSYNC_SSH}" \
  scripts/coarsen_geo_patches_to_budget.py \
  scripts/optimize_geo_patch_merges.py \
  scripts/qa_object_voxel_overlap.py \
  scripts/analyze_geo_patch_bbox_overlap.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"

"${SSH[@]}" bash -s <<REMOTE
set -euo pipefail
mkdir -p "${OUTPUT_DIR}" "${REMOTE_WORK}/run_scripts"
cat > "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd "${REMOTE_REPO}"
source "${REMOTE_VENV}/bin/activate"
OUT="${OUTPUT_DIR}"
REGION="${REGION_RUN}"
mkdir -p "\${OUT}"
printf '%s\n' "\${OUT}" > "${REMOTE_WORK}/latest_geo_patch_5070_coarsen.txt"

"${REMOTE_PYTHON}" scripts/coarsen_geo_patches_to_budget.py \
  --region-input "\${REGION}/_cpp_region_grower_input.bin" \
  --labels "\${REGION}/_cpp_region_grower_labels.bin" \
  --output-dir "\${OUT}" \
  --target-patches "${TARGET_PATCHES}" \
  --noise-patch-voxels "${NOISE_PATCH_VOXELS}" \
  --precollapse-mode "${PRECOLLAPSE_MODE}" \
  --precollapse-grid-size "${PRECOLLAPSE_GRID_SIZE}" \
  --precollapse-min-component-voxels "${PRECOLLAPSE_MIN_COMPONENT_VOXELS}" \
  --neighbors-per-patch "${NEIGHBORS_PER_PATCH}" \
  --max-centroid-distance "${MAX_CENTROID_DISTANCE}" \
  --max-bbox-gap "${MAX_BBOX_GAP}" \
  --min-merge-score "${MIN_MERGE_SCORE}" \
  --max-color-distance "${MAX_COLOR_DISTANCE}" \
  --hard-color-distance "${HARD_COLOR_DISTANCE}" \
  --max-component-voxels "${MAX_COMPONENT_VOXELS}" \
  --preview-stride "${PREVIEW_STRIDE}" \
  --overlap-voxel-size "${OVERLAP_VOXEL_SIZE}" \
  --overlap-merge-passes 1 \
  --overlap-min-ratio "${OVERLAP_MIN_RATIO}" \
  --overlap-min-score "${OVERLAP_MIN_SCORE}" \
  --overlap-block-stable-mismatch \
  --overlap-block-stable-rough \
  > "\${OUT}/coarsen.log" 2>&1

"${REMOTE_PYTHON}" scripts/qa_object_voxel_overlap.py \
  --ply "\${OUT}/geo_patches_coarse_stride${PREVIEW_STRIDE}.ply" \
  --voxel-size 0.20 \
  --max-pairs 100 \
  --output-json "\${OUT}/voxel_overlap_020_report.json" \
  --summary-only > "\${OUT}/qa_overlap_020.log" 2>&1

"${REMOTE_PYTHON}" scripts/analyze_geo_patch_bbox_overlap.py \
  --input-jsonl "\${OUT}/geo_patches_coarse.jsonl" \
  --output-dir "\${OUT}/bbox_overlap_top1000" \
  --top-n 1000 \
  --bbox-pad 0.05 \
  > "\${OUT}/bbox_overlap_top1000.log" 2>&1

date -Is > "\${OUT}/DONE"
SCRIPT
chmod +x "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux ls
REMOTE
