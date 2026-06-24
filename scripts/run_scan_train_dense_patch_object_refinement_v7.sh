#!/usr/bin/env bash
set -euo pipefail

# Run dense Patch object-refinement v7 on the current 4090D host.
#
# This uses the dense 0.03m Opt-LAS chain that is actually present on
# scan-train.  It starts from r4 region labels optimized by attachment/contact
# evidence, then proposes structural/multimaterial object candidates and builds
# conservative object labels.  It does not use semantic labels as merge input.

REMOTE_HOST="${REMOTE_HOST:-scan-train}"
REMOTE_REPO="${REMOTE_REPO:-/root/epfs/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
TMUX_SESSION="${TMUX_SESSION:-scan_dense_object_v7}"
RUN="${RUN:-0}"

BASE="${BASE:-${REMOTE_WORK}/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623}"
REGION_INPUT="${REGION_INPUT:-${BASE}/_cpp_region_grower_input.bin}"
PATCH_LABELS="${PATCH_LABELS:-${BASE}/energy_attach_v4_contact_evidence/geo_patches_energy_attach_v4_contact_evidence_labels.bin}"
OUT_NAME="${OUT_NAME:-dense_patch_object_refinement_v7_r4_attach_v4_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE}/${OUT_NAME}}"

PYTHON="${PYTHON:-python3}"
EDGE_SOURCE="${EDGE_SOURCE:-region}"
PREVIEW_STRIDE="${PREVIEW_STRIDE:-10}"

echo "remote=${REMOTE_HOST}"
echo "region_input=${REGION_INPUT}"
echo "patch_labels=${PATCH_LABELS}"
echo "output_dir=${OUTPUT_DIR}"
if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to launch tmux session ${TMUX_SESSION}"
  exit 0
fi

rsync -az \
  scripts/propose_geo_patch_object_merges.py \
  scripts/build_geo_patch_objects_from_candidates.py \
  scripts/run_dense_patch_object_refinement_v7.py \
  scripts/optimize_patch_graph_energy.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"

ssh "${REMOTE_HOST}" bash -s <<REMOTE
set -euo pipefail
test -f "${REGION_INPUT}"
test -f "${PATCH_LABELS}"
mkdir -p "${OUTPUT_DIR}" "${REMOTE_WORK}/run_scripts"
cat > "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd "${REMOTE_REPO}"
${PYTHON} scripts/run_dense_patch_object_refinement_v7.py \
  --region-input "${REGION_INPUT}" \
  --patch-labels "${PATCH_LABELS}" \
  --output-dir "${OUTPUT_DIR}" \
  --python "${PYTHON}" \
  --edge-source "${EDGE_SOURCE}" \
  --preview-stride "${PREVIEW_STRIDE}" \
  --run \
  > "${OUTPUT_DIR}/run_dense_patch_object_refinement_v7.log" 2>&1
date -Is > "${OUTPUT_DIR}/DONE"
SCRIPT
chmod +x "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux ls
REMOTE
