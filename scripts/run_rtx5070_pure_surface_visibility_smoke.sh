#!/usr/bin/env bash
set -euo pipefail

# Run the clean structural-region + first-touch target attachment smoke on
# scan-rtx5070.  The heavy PCD/target work stays on the 5070Ti host; this script
# only syncs the small source files and the reusable drivability prior if needed.

REMOTE_HOST="${REMOTE_HOST:-scan-rtx5070}"
REMOTE_REPO="${REMOTE_REPO:-/home/zsh/Work/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
REMOTE_VENV="${REMOTE_VENV:-/home/zsh/Work/SCAN/.venvs/scan-semantic}"

LOCAL_DRIVABILITY_PCD="${LOCAL_DRIVABILITY_PCD:-/Users/skkac/Work/SCAN/drivability_cpp/output/MT20260616-175807_drivable_points_collision_arm64_wallbfs.pcd}"
REMOTE_DRIVABILITY_PCD="${REMOTE_DRIVABILITY_PCD:-${REMOTE_WORK}/structural_priors/MT20260616-175807_drivable_points_collision_arm64_wallbfs.pcd}"

TARGET_DIR="${TARGET_DIR:-${REMOTE_WORK}/frame_targets_probe_3400_3500_baseline}"
TARGET_JSONL="${TARGET_JSONL:-${TARGET_DIR}/frame_targets.jsonl}"
TARGET_PLY="${TARGET_PLY:-${TARGET_DIR}/frame_targets.ply}"
OUT_DIR="${OUT_DIR:-${REMOTE_WORK}/pure_surface_visibility_smoke_3400_3500}"
STRUCTURAL_VOXEL_SIZE="${STRUCTURAL_VOXEL_SIZE:-0.10}"

if [[ ! -f "${LOCAL_DRIVABILITY_PCD}" ]]; then
  echo "missing local drivability PCD: ${LOCAL_DRIVABILITY_PCD}" >&2
  exit 1
fi

echo "[sync] scripts -> ${REMOTE_HOST}:${REMOTE_REPO}/scripts"
rsync -az \
  scripts/build_structural_region_field.py \
  scripts/classify_surface_attachment.py \
  scripts/fuse_targets_to_objects.py \
  scripts/export_frame_target_objects_for_viewer.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"

echo "[sync] drivability prior -> ${REMOTE_HOST}:${REMOTE_DRIVABILITY_PCD}"
ssh "${REMOTE_HOST}" "mkdir -p '$(dirname "${REMOTE_DRIVABILITY_PCD}")'"
rsync -az --ignore-existing "${LOCAL_DRIVABILITY_PCD}" "${REMOTE_HOST}:${REMOTE_DRIVABILITY_PCD}"

ssh "${REMOTE_HOST}" "cd '${REMOTE_REPO}' && bash -lc '
set -euo pipefail
source '${REMOTE_VENV}'/bin/activate
rm -rf '${OUT_DIR}'
mkdir -p '${OUT_DIR}'
python scripts/build_structural_region_field.py \
  --drivability-pcd '${REMOTE_DRIVABILITY_PCD}' \
  --output-npz '${OUT_DIR}/structural_region_field.npz' \
  --report '${OUT_DIR}/structural_region_field_report.json' \
  --voxel-size '${STRUCTURAL_VOXEL_SIZE}'
python scripts/classify_surface_attachment.py \
  --targets-jsonl '${TARGET_JSONL}' \
  --target-ply '${TARGET_PLY}' \
  --structural-field '${OUT_DIR}/structural_region_field.npz' \
  --output-jsonl '${OUT_DIR}/targets_surface_attachment.jsonl' \
  --report '${OUT_DIR}/surface_attachment_report.json'
python scripts/fuse_targets_to_objects.py \
  --targets '${OUT_DIR}/targets_surface_attachment.jsonl' \
  --output-dir '${OUT_DIR}/objects' \
  --strict-surface-labels
python scripts/export_frame_target_objects_for_viewer.py \
  --targets-jsonl '${OUT_DIR}/targets_surface_attachment.jsonl' \
  --target-ply '${TARGET_PLY}' \
  --objects-jsonl '${OUT_DIR}/objects/objects.jsonl' \
  --output-dir '${OUT_DIR}/viewer' \
  --stride 1 \
  --keep-target-list
echo DONE:${OUT_DIR}
'"

echo "[result] ${REMOTE_HOST}:${OUT_DIR}"
