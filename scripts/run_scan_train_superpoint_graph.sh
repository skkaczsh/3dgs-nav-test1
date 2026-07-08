#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-scan-train}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_REPO="${REMOTE_REPO:-/root/epfs/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
BASE="${BASE:-${REMOTE_WORK}/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623}"
REGION_INPUT="${REGION_INPUT:-${BASE}/_cpp_region_grower_input.bin}"
PATCH_LABELS="${PATCH_LABELS:-${BASE}/energy_attach_v4_contact_evidence/geo_patches_energy_attach_v4_contact_evidence_labels.bin}"
OUT_NAME="${OUT_NAME:-superpoint_graph_v1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE}/${OUT_NAME}}"
TMUX_SESSION="${TMUX_SESSION:-scan_superpoint_graph}"
RUN="${RUN:-0}"
PYTHON="${PYTHON:-python3}"
MIN_EDGE_SCORE="${MIN_EDGE_SCORE:-0.78}"
MAX_MERGED_ENTROPY="${MAX_MERGED_ENTROPY:-1.20}"
FH_K="${FH_K:-0}"
ENABLE_UNCERTAIN="${ENABLE_UNCERTAIN:-0}"
EXTERNAL_EDGE_EVIDENCE="${EXTERNAL_EDGE_EVIDENCE:-}"
EXTERNAL_EDGE_WEIGHT="${EXTERNAL_EDGE_WEIGHT:-0.15}"

echo "remote=${REMOTE_HOST}"
echo "region_input=${REGION_INPUT}"
echo "patch_labels=${PATCH_LABELS}"
echo "output_dir=${OUTPUT_DIR}"
echo "min_edge_score=${MIN_EDGE_SCORE}"
echo "max_merged_entropy=${MAX_MERGED_ENTROPY}"
echo "fh_k=${FH_K}"
echo "enable_uncertain=${ENABLE_UNCERTAIN}"
echo "external_edge_evidence=${EXTERNAL_EDGE_EVIDENCE:-none}"
echo "external_edge_weight=${EXTERNAL_EDGE_WEIGHT}"

if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to launch tmux session ${TMUX_SESSION}"
  exit 0
fi

rsync -az \
  "${LOCAL_REPO}/scripts/cluster_superpoint_graph.py" \
  "${LOCAL_REPO}/scripts/optimize_patch_graph_energy.py" \
  "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"

ssh "${REMOTE_HOST}" bash -s <<REMOTE
set -euo pipefail
test -f "${REGION_INPUT}"
test -f "${PATCH_LABELS}"
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/run.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd "${REMOTE_REPO}"
UNCERTAIN_ARGS=""
if [[ "${ENABLE_UNCERTAIN}" == "1" ]]; then
  UNCERTAIN_ARGS="
    --enable-uncertain-fragment-candidates
    --uncertain-min-stable-voxels 10000
    --uncertain-max-fragment-voxels 5000
    --uncertain-min-contact-points 16
    --uncertain-max-color-distance 75
    --uncertain-max-stable-patches 200
  "
fi
EXTERNAL_ARGS=""
if [[ -n "${EXTERNAL_EDGE_EVIDENCE}" ]]; then
  EXTERNAL_ARGS="
    --external-edge-evidence ${EXTERNAL_EDGE_EVIDENCE}
    --external-edge-weight ${EXTERNAL_EDGE_WEIGHT}
  "
fi
"${PYTHON}" scripts/cluster_superpoint_graph.py \
  --region-input "${REGION_INPUT}" \
  --labels "${PATCH_LABELS}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-stem superpoint_graph_v1 \
  --min-edge-score "${MIN_EDGE_SCORE}" \
  --max-merged-entropy "${MAX_MERGED_ENTROPY}" \
  --fh-k "${FH_K}" \
  --max-color-distance 90 \
  --enable-structural-merge-veto \
  --structural-veto-min-voxels 1000 \
  --preview-stride 10 \
  \${UNCERTAIN_ARGS} \
  \${EXTERNAL_ARGS} \
  > "${OUTPUT_DIR}/cluster.log" 2>&1
date -Is > "${OUTPUT_DIR}/DONE"
SCRIPT
chmod +x "${OUTPUT_DIR}/run.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${OUTPUT_DIR}/run.sh"
tmux ls
REMOTE
