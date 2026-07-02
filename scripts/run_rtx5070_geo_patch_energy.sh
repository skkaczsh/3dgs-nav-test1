#!/usr/bin/env bash
set -euo pipefail

# Full 5070Ti GeoPatch region-energy route.
#
# This runner is intentionally separate from run_rtx5070_geo_patch_route.sh:
# it tests the geometry Patch layer only, not semantic object classification.

REMOTE_HOST="${REMOTE_HOST:-zsh@skkac.top}"
REMOTE_PORT="${REMOTE_PORT:-6010}"
REMOTE_KEY="${REMOTE_KEY:-/Users/skkac/.ssh/id_ed25519}"
REMOTE_REPO="${REMOTE_REPO:-/home/zsh/Work/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
REMOTE_VENV="${REMOTE_VENV:-/home/zsh/Work/SCAN/.venvs/scan-semantic}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"
TMUX_SESSION="${TMUX_SESSION:-scan_patch_energy_v3_003}"

RUN="${RUN:-0}"
REQUIRE_CURRENT_DENSE_INPUTS="${REQUIRE_CURRENT_DENSE_INPUTS:-1}"
KILL_LLAMA="${KILL_LLAMA:-1}"
OUT_NAME="${OUT_NAME:-geo_patch_5070_full_0000_6180_energy_v3_voxel003_20260624_0018}"
INPUT_PLY="${INPUT_PLY:-${REMOTE_WORK}/dense_sources/dense_las_voxel003_20260624/dense_las_voxel003_binary.ply}"
OUTPUT_DIR="${OUTPUT_DIR:-${REMOTE_WORK}/${OUT_NAME}}"

VOXEL_SIZE="${VOXEL_SIZE:-0.03}"
FEATURE_RADIUS_VOXELS="${FEATURE_RADIUS_VOXELS:-3}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-8192}"
CONNECT_RADIUS_VOXELS="${CONNECT_RADIUS_VOXELS:-2}"
MIN_EDGE_SCORE="${MIN_EDGE_SCORE:-0.46}"
MAX_COLOR_DISTANCE="${MAX_COLOR_DISTANCE:-150}"
PREVIEW_STRIDE="${PREVIEW_STRIDE:-3}"
MAX_ITERS="${MAX_ITERS:-6}"
MIN_MERGE_GAIN="${MIN_MERGE_GAIN:-0.30}"
MERGE_MIN_NEIGHBOR_SUPPORT="${MERGE_MIN_NEIGHBOR_SUPPORT:-0.05}"
MAX_MERGE_CANDIDATES="${MAX_MERGE_CANDIDATES:-240000}"

SSH=(ssh -i "${REMOTE_KEY}" -p "${REMOTE_PORT}" "${REMOTE_HOST}")
RSYNC_SSH="ssh -i ${REMOTE_KEY} -p ${REMOTE_PORT}"

if [[ "${REQUIRE_CURRENT_DENSE_INPUTS}" == "1" ]]; then
  "${LOCAL_PYTHON}" scripts/validate_production_inputs.py --require-current-dense "${INPUT_PLY}"
else
  "${LOCAL_PYTHON}" scripts/validate_production_inputs.py "${INPUT_PLY}"
fi

echo "remote=${REMOTE_HOST}:${REMOTE_PORT}"
echo "input_ply=${INPUT_PLY}"
echo "output_dir=${OUTPUT_DIR}"
if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

rsync -az -e "${RSYNC_SSH}" \
  scripts/build_geo_patch_region_model.py \
  scripts/optimize_patch_graph_energy.py \
  scripts/qa_object_voxel_overlap.py \
  scripts/analyze_geo_patch_bbox_overlap.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"

"${SSH[@]}" bash -s <<REMOTE
set -euo pipefail
if [[ "${KILL_LLAMA}" == "1" ]]; then
  pkill -f "/home/zsh/llama-cpp-turboquant/build/bin/llama-server" || true
fi
mkdir -p "${OUTPUT_DIR}/energy" "${REMOTE_WORK}/run_scripts"
cat > "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd "${REMOTE_REPO}"
source "${REMOTE_VENV}/bin/activate"
OUT="${OUTPUT_DIR}"
INPUT="${INPUT_PLY}"
mkdir -p "\${OUT}/energy"
printf '%s\n' "\${OUT}" > "${REMOTE_WORK}/latest_geo_patch_5070_energy_v3.txt"

"${REMOTE_PYTHON}" scripts/build_geo_patch_region_model.py \
  --input-ply "\${INPUT}" \
  --output-dir "\${OUT}" \
  --voxel-size "${VOXEL_SIZE}" \
  --voxel-backend torch \
  --feature-backend torch \
  --feature-radius-voxels "${FEATURE_RADIUS_VOXELS}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE}" \
  --torch-device cuda:0 \
  --region-grow-backend cpp \
  --connect-radius-voxels "${CONNECT_RADIUS_VOXELS}" \
  --min-edge-score "${MIN_EDGE_SCORE}" \
  --max-color-distance "${MAX_COLOR_DISTANCE}" \
  --bucket-guard same-bucket-or-fine-color \
  > "\${OUT}/build.log" 2>&1

"${REMOTE_PYTHON}" scripts/optimize_patch_graph_energy.py \
  --region-input "\${OUT}/_cpp_region_grower_input.bin" \
  --labels "\${OUT}/_cpp_region_grower_labels.bin" \
  --output-dir "\${OUT}/energy" \
  --output-stem geo_patches_energy_v3 \
  --enable-split --enable-boundary --enable-annealing \
  --max-iters "${MAX_ITERS}" \
  --preview-stride "${PREVIEW_STRIDE}" \
  --max-color-distance "${MAX_COLOR_DISTANCE}" \
  --min-merge-gain "${MIN_MERGE_GAIN}" \
  --merge-min-neighbor-support "${MERGE_MIN_NEIGHBOR_SUPPORT}" \
  --max-merge-candidates "${MAX_MERGE_CANDIDATES}" \
  > "\${OUT}/energy/optimize.log" 2>&1

"${REMOTE_PYTHON}" scripts/qa_object_voxel_overlap.py \
  --ply "\${OUT}/energy/geo_patches_energy_v3_stride${PREVIEW_STRIDE}.ply" \
  --voxel-size 0.20 \
  --max-pairs 100 \
  --output-json "\${OUT}/energy/voxel_overlap_020_report.json" \
  --summary-only > "\${OUT}/energy/qa_overlap_020.log" 2>&1

"${REMOTE_PYTHON}" scripts/analyze_geo_patch_bbox_overlap.py \
  --input-jsonl "\${OUT}/energy/geo_patches_energy_v3.jsonl" \
  --output-dir "\${OUT}/energy/bbox_overlap_top1000" \
  --top-n 1000 \
  --bbox-pad 0.05 \
  > "\${OUT}/energy/bbox_overlap_top1000.log" 2>&1

date -Is > "\${OUT}/DONE"
SCRIPT
chmod +x "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux ls
REMOTE
