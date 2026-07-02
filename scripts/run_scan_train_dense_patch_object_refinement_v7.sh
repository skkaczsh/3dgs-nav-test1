#!/usr/bin/env bash
set -euo pipefail

# Run dense Patch object-refinement v7 on the current 4090D host.
#
# This uses the dense 0.03m Opt-LAS chain that is actually present on
# scan-train.  It starts from r4 region labels optimized by attachment/contact
# evidence, then proposes structural/multimaterial object candidates and builds
# conservative object labels.  It does not use semantic labels as merge input.

REMOTE_HOST="${REMOTE_HOST:-scan-train}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_REPO="${REMOTE_REPO:-/root/epfs/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
TMUX_SESSION="${TMUX_SESSION:-scan_dense_object_v7}"
RUN="${RUN:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
REQUIRE_CURRENT_DENSE_INPUTS="${REQUIRE_CURRENT_DENSE_INPUTS:-1}"
PREFLIGHT="${PREFLIGHT:-${LOCAL_REPO}/scripts/validate_current_mainline.py}"

BASE="${BASE:-${REMOTE_WORK}/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623}"
REGION_INPUT="${REGION_INPUT:-${BASE}/_cpp_region_grower_input.bin}"
PATCH_LABELS="${PATCH_LABELS:-${BASE}/energy_attach_v4_contact_evidence/geo_patches_energy_attach_v4_contact_evidence_labels.bin}"
OUT_NAME="${OUT_NAME:-dense_patch_object_refinement_v7_r4_attach_v4_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE}/${OUT_NAME}}"

PYTHON="${PYTHON:-python3}"
EDGE_SOURCE="${EDGE_SOURCE:-region}"
PREVIEW_STRIDE="${PREVIEW_STRIDE:-10}"

# Candidate recall knobs.  Keep these high-recall and let the object builder
# decide with stricter gates.
MIN_PATCH_VOXELS="${MIN_PATCH_VOXELS:-40}"
MIN_SHARED_EDGES="${MIN_SHARED_EDGES:-3}"
MIN_CONTACT_RATIO="${MIN_CONTACT_RATIO:-0.006}"
MAX_BBOX_GAP="${MAX_BBOX_GAP:-0.20}"
MAX_COLOR_DISTANCE="${MAX_COLOR_DISTANCE:-105}"
MIN_NORMAL_SCORE="${MIN_NORMAL_SCORE:-0.42}"
MIN_BUCKET_SCORE="${MIN_BUCKET_SCORE:-0.42}"
MIN_SCORE="${MIN_SCORE:-0.54}"
CONTACT_RATIO_NORM="${CONTACT_RATIO_NORM:-0.18}"
MAX_CANDIDATES="${MAX_CANDIDATES:-50000}"
MIN_STRUCTURAL_SCORE="${MIN_STRUCTURAL_SCORE:-0.70}"
STRUCTURAL_MIN_CONTACT_RATIO="${STRUCTURAL_MIN_CONTACT_RATIO:-0.025}"
STRUCTURAL_MIN_SHARED_EDGES="${STRUCTURAL_MIN_SHARED_EDGES:-12}"
STRUCTURAL_MIN_NORMAL_SCORE="${STRUCTURAL_MIN_NORMAL_SCORE:-0.56}"
STRUCTURAL_MAX_BBOX_GAP="${STRUCTURAL_MAX_BBOX_GAP:-0.10}"

# Acceptance knobs.  These should stay conservative to avoid over-merge.
ACCEPT_MIN_SCORE="${ACCEPT_MIN_SCORE:-0.80}"
ACCEPT_MIN_CONTACT_RATIO="${ACCEPT_MIN_CONTACT_RATIO:-0.08}"
ACCEPT_MIN_SHARED_EDGES="${ACCEPT_MIN_SHARED_EDGES:-32}"
ACCEPT_MAX_COLOR_DISTANCE="${ACCEPT_MAX_COLOR_DISTANCE:-55}"
ACCEPT_MAX_BBOX_GAP="${ACCEPT_MAX_BBOX_GAP:-0.08}"
ACCEPT_MIN_NORMAL_SCORE="${ACCEPT_MIN_NORMAL_SCORE:-0.65}"
ACCEPT_MIN_STRUCTURAL_SCORE="${ACCEPT_MIN_STRUCTURAL_SCORE:-0.74}"
ACCEPT_STRUCTURAL_MIN_CONTACT_RATIO="${ACCEPT_STRUCTURAL_MIN_CONTACT_RATIO:-0.035}"
ACCEPT_STRUCTURAL_MIN_SHARED_EDGES="${ACCEPT_STRUCTURAL_MIN_SHARED_EDGES:-24}"
ACCEPT_STRUCTURAL_MIN_NORMAL_SCORE="${ACCEPT_STRUCTURAL_MIN_NORMAL_SCORE:-0.58}"
ACCEPT_STRUCTURAL_MAX_BBOX_GAP="${ACCEPT_STRUCTURAL_MAX_BBOX_GAP:-0.08}"
ATTACHMENT_MIN_SCORE="${ATTACHMENT_MIN_SCORE:-0.82}"
ATTACHMENT_MIN_CONTACT_RATIO="${ATTACHMENT_MIN_CONTACT_RATIO:-0.16}"
ATTACHMENT_MIN_SHARED_EDGES="${ATTACHMENT_MIN_SHARED_EDGES:-48}"
ATTACHMENT_MAX_COLOR_DISTANCE="${ATTACHMENT_MAX_COLOR_DISTANCE:-38}"
ATTACHMENT_MIN_NORMAL_SCORE="${ATTACHMENT_MIN_NORMAL_SCORE:-0.65}"
ATTACHMENT_MAX_BBOX_GAP="${ATTACHMENT_MAX_BBOX_GAP:-0.06}"
ATTACHMENT_MAX_FRAGMENT_VOXELS="${ATTACHMENT_MAX_FRAGMENT_VOXELS:-1200}"
ATTACHMENT_MIN_ANCHOR_VOXELS="${ATTACHMENT_MIN_ANCHOR_VOXELS:-100000}"
ATTACHMENT_MIN_SIZE_RATIO="${ATTACHMENT_MIN_SIZE_RATIO:-500}"

echo "remote=${REMOTE_HOST}"
echo "region_input=${REGION_INPUT}"
echo "patch_labels=${PATCH_LABELS}"
echo "output_dir=${OUTPUT_DIR}"
if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to launch tmux session ${TMUX_SESSION}"
  exit 0
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  echo "preflight=${PREFLIGHT}"
  "${PYTHON}" "${PREFLIGHT}"
  if [[ "${REQUIRE_CURRENT_DENSE_INPUTS}" == "1" ]]; then
    "${PYTHON}" "${LOCAL_REPO}/scripts/validate_production_inputs.py" --require-current-dense "${REGION_INPUT}" "${PATCH_LABELS}"
  else
    "${PYTHON}" "${LOCAL_REPO}/scripts/validate_production_inputs.py" "${REGION_INPUT}" "${PATCH_LABELS}"
  fi
else
  echo "preflight=skipped"
fi

rsync -az \
  scripts/current_mainline_contract.py \
  scripts/validate_production_inputs.py \
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
  --min-patch-voxels "${MIN_PATCH_VOXELS}" \
  --min-shared-edges "${MIN_SHARED_EDGES}" \
  --min-contact-ratio "${MIN_CONTACT_RATIO}" \
  --max-bbox-gap "${MAX_BBOX_GAP}" \
  --max-color-distance "${MAX_COLOR_DISTANCE}" \
  --min-normal-score "${MIN_NORMAL_SCORE}" \
  --min-bucket-score "${MIN_BUCKET_SCORE}" \
  --min-score "${MIN_SCORE}" \
  --contact-ratio-norm "${CONTACT_RATIO_NORM}" \
  --max-candidates "${MAX_CANDIDATES}" \
  --min-structural-score "${MIN_STRUCTURAL_SCORE}" \
  --structural-min-contact-ratio "${STRUCTURAL_MIN_CONTACT_RATIO}" \
  --structural-min-shared-edges "${STRUCTURAL_MIN_SHARED_EDGES}" \
  --structural-min-normal-score "${STRUCTURAL_MIN_NORMAL_SCORE}" \
  --structural-max-bbox-gap "${STRUCTURAL_MAX_BBOX_GAP}" \
  --accept-min-score "${ACCEPT_MIN_SCORE}" \
  --accept-min-contact-ratio "${ACCEPT_MIN_CONTACT_RATIO}" \
  --accept-min-shared-edges "${ACCEPT_MIN_SHARED_EDGES}" \
  --accept-max-color-distance "${ACCEPT_MAX_COLOR_DISTANCE}" \
  --accept-max-bbox-gap "${ACCEPT_MAX_BBOX_GAP}" \
  --accept-min-normal-score "${ACCEPT_MIN_NORMAL_SCORE}" \
  --accept-min-structural-score "${ACCEPT_MIN_STRUCTURAL_SCORE}" \
  --accept-structural-min-contact-ratio "${ACCEPT_STRUCTURAL_MIN_CONTACT_RATIO}" \
  --accept-structural-min-shared-edges "${ACCEPT_STRUCTURAL_MIN_SHARED_EDGES}" \
  --accept-structural-min-normal-score "${ACCEPT_STRUCTURAL_MIN_NORMAL_SCORE}" \
  --accept-structural-max-bbox-gap "${ACCEPT_STRUCTURAL_MAX_BBOX_GAP}" \
  --attachment-min-score "${ATTACHMENT_MIN_SCORE}" \
  --attachment-min-contact-ratio "${ATTACHMENT_MIN_CONTACT_RATIO}" \
  --attachment-min-shared-edges "${ATTACHMENT_MIN_SHARED_EDGES}" \
  --attachment-max-color-distance "${ATTACHMENT_MAX_COLOR_DISTANCE}" \
  --attachment-min-normal-score "${ATTACHMENT_MIN_NORMAL_SCORE}" \
  --attachment-max-bbox-gap "${ATTACHMENT_MAX_BBOX_GAP}" \
  --attachment-max-fragment-voxels "${ATTACHMENT_MAX_FRAGMENT_VOXELS}" \
  --attachment-min-anchor-voxels "${ATTACHMENT_MIN_ANCHOR_VOXELS}" \
  --attachment-min-size-ratio "${ATTACHMENT_MIN_SIZE_RATIO}" \
  --skip-mainline-healthcheck \
  --run \
  > "${OUTPUT_DIR}/run_dense_patch_object_refinement_v7.log" 2>&1
date -Is > "${OUTPUT_DIR}/DONE"
SCRIPT
chmod +x "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux ls
REMOTE
