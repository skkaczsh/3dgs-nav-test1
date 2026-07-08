#!/usr/bin/env bash
set -euo pipefail

# Re-run the 4090D dense LAS patch optimizer with bucket-connectivity splitting
# plus attachment absorption.  This is an evaluation route, not a promoted
# semantic baseline.

REMOTE_HOST="${REMOTE_HOST:-scan-train}"
REMOTE_REPO="${REMOTE_REPO:-/root/epfs/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
BASE="${BASE:-${REMOTE_WORK}/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623}"
REGION_INPUT="${REGION_INPUT:-${BASE}/_cpp_region_grower_input.bin}"
PATCH_LABELS="${PATCH_LABELS:-${BASE}/energy_attach_v4_contact_evidence/geo_patches_energy_attach_v4_contact_evidence_labels.bin}"
OUT_NAME="${OUT_NAME:-energy_bucket_split_attach_v2_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE}/${OUT_NAME}}"
TMUX_SESSION="${TMUX_SESSION:-scan_patch_bucket_attach_eval}"
RUN="${RUN:-0}"
PYTHON="${PYTHON:-python3}"
ENABLE_STRUCTURAL_VETO="${ENABLE_STRUCTURAL_VETO:-0}"
STRUCTURAL_VETO_MIN_BUCKET_RATIO="${STRUCTURAL_VETO_MIN_BUCKET_RATIO:-0.20}"
STRUCTURAL_VETO_MIN_VOXELS="${STRUCTURAL_VETO_MIN_VOXELS:-1000}"

echo "remote=${REMOTE_HOST}"
echo "region_input=${REGION_INPUT}"
echo "patch_labels=${PATCH_LABELS}"
echo "output_dir=${OUTPUT_DIR}"
echo "enable_structural_veto=${ENABLE_STRUCTURAL_VETO}"

STRUCTURAL_ARGS_TEXT=""
if [[ "${ENABLE_STRUCTURAL_VETO}" == "1" ]]; then
  STRUCTURAL_ARGS_TEXT="--enable-structural-merge-veto --structural-veto-min-bucket-ratio ${STRUCTURAL_VETO_MIN_BUCKET_RATIO} --structural-veto-min-voxels ${STRUCTURAL_VETO_MIN_VOXELS}"
fi

if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to launch tmux session ${TMUX_SESSION}"
  exit 0
fi

rsync -az scripts/optimize_patch_graph_energy.py "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"

ssh "${REMOTE_HOST}" bash -s <<REMOTE
set -euo pipefail
test -f "${REGION_INPUT}"
test -f "${PATCH_LABELS}"
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/run.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd "${REMOTE_REPO}"
mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" scripts/optimize_patch_graph_energy.py \
  --region-input "${REGION_INPUT}" \
  --labels "${PATCH_LABELS}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-stem geo_patches_bucket_split_attach_v2 \
  --enable-split --enable-boundary --enable-annealing \
  --enable-bucket-connectivity-split \
  --bucket-split-min-voxels 1400 \
  --bucket-split-min-bucket-ratio 0.15 \
  --bucket-split-target-buckets unknown,thin_linear,rough_mixed \
  --enable-attachment-merge \
  --attachment-min-score 0.78 \
  --attachment-min-contact-ratio 0.08 \
  --attachment-min-shared-edges 8 \
  --attachment-max-color-distance 70 \
  --attachment-min-normal-score 0.42 \
  --attachment-max-bbox-gap 0.08 \
  --attachment-max-fragment-voxels 5000 \
  --attachment-min-anchor-voxels 20000 \
  --attachment-min-size-ratio 4 \
  --attachment-contact-norm 0.18 \
  --attachment-color-weight 0.32 \
  --attachment-normal-weight 0.16 \
  --attachment-bucket-weight 0.14 \
  --attachment-contact-weight 0.30 \
  --attachment-gap-weight 0.08 \
  --max-iters 4 \
  --preview-stride 10 \
  --max-color-distance 150 \
  --min-merge-gain 0.30 \
  --merge-min-neighbor-support 0.05 \
  --max-merge-candidates 240000 \
  ${STRUCTURAL_ARGS_TEXT} \
  > "${OUTPUT_DIR}/optimize.log" 2>&1
date -Is > "${OUTPUT_DIR}/DONE"
SCRIPT
chmod +x "${OUTPUT_DIR}/run.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${OUTPUT_DIR}/run.sh"
tmux ls
REMOTE
