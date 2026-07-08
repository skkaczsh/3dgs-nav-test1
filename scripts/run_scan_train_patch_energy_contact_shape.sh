#!/usr/bin/env bash
set -euo pipefail

# Run the 4090D dense 0.03m Opt-LAS patch-energy experiment with contact-shape
# attachment scoring.  This is a patch-layer validation, not semantic fusion.

REMOTE_HOST="${REMOTE_HOST:-scan-train}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_REPO="${REMOTE_REPO:-/root/epfs/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
BASE="${BASE:-${REMOTE_WORK}/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623}"
REGION_INPUT="${REGION_INPUT:-${BASE}/_cpp_region_grower_input.bin}"
PATCH_LABELS="${PATCH_LABELS:-${BASE}/energy_attach_v4_contact_evidence/geo_patches_energy_attach_v4_contact_evidence_labels.bin}"
OUT_NAME="${OUT_NAME:-energy_attachment_shape_v1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE}/${OUT_NAME}}"
TMUX_SESSION="${TMUX_SESSION:-scan_patch_energy_contact_shape}"
PYTHON="${PYTHON:-python3}"
RUN="${RUN:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

echo "remote=${REMOTE_HOST}"
echo "region_input=${REGION_INPUT}"
echo "patch_labels=${PATCH_LABELS}"
echo "output_dir=${OUTPUT_DIR}"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON}" "${LOCAL_REPO}/scripts/validate_current_mainline.py" >/dev/null
  "${PYTHON}" "${LOCAL_REPO}/scripts/validate_production_inputs.py" --require-current-dense "${REGION_INPUT}" "${PATCH_LABELS}" >/dev/null
fi

if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to launch tmux session ${TMUX_SESSION}"
  exit 0
fi

rsync -az \
  "${LOCAL_REPO}/scripts/optimize_patch_graph_energy.py" \
  "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"
rsync -az "${LOCAL_REPO}/docs/current_dense_patch_state.json" "${REMOTE_HOST}:${REMOTE_REPO}/docs/"

ssh "${REMOTE_HOST}" bash -s <<REMOTE
set -euo pipefail
test -f "${REGION_INPUT}"
test -f "${PATCH_LABELS}"
mkdir -p "${OUTPUT_DIR}" "${REMOTE_WORK}/run_scripts"
cat > "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd "${REMOTE_REPO}"
"${PYTHON}" scripts/optimize_patch_graph_energy.py \
  --region-input "${REGION_INPUT}" \
  --labels "${PATCH_LABELS}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-stem geo_patches_energy_attachment_shape_v1 \
  --max-iters 1 \
  --min-anchor-voxels 900 \
  --enable-attachment-merge \
  --enable-fragment-evidence-attachment \
  --enable-fh-merge-guard \
  --fh-k 55 \
  --fh-color-weight 0.55 \
  --fh-color-p90-weight 0.15 \
  --fh-normal-weight 0.20 \
  --fh-shape-weight 0.10 \
  --attachment-shape-weight 0.15 \
  --min-merge-gain 0.35 \
  --preview-stride 10 \
  > "${OUTPUT_DIR}/optimize.log" 2>&1
date -Is > "${OUTPUT_DIR}/DONE"
SCRIPT
chmod +x "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${REMOTE_WORK}/run_scripts/${OUT_NAME}.sh"
tmux ls
REMOTE
