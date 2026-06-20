#!/usr/bin/env bash
set -euo pipefail

# Production entry for the clean parking-lot surface-visibility route on
# scan-rtx5070.
#
# The route intentionally reuses the validated geometry/mask target builder up
# to frame-local Target generation, then switches to the non-semantic
# drivability structural field and attachment-aware Object fusion.
#
# Default mode is a dry run.  Use RUN=1 to execute on the remote host.  Existing
# output directories are protected by the underlying route unless OVERWRITE=1.

REMOTE_HOST="${REMOTE_HOST:-scan-rtx5070}"
REMOTE_REPO="${REMOTE_REPO:-/home/zsh/Work/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
REMOTE_VENV="${REMOTE_VENV:-/home/zsh/Work/SCAN/.venvs/scan-semantic}"

RUN="${RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
START="${START:-3400}"
END="${END:-3500}"
STRIDE="${STRIDE:-10}"
CAMS="${CAMS:-0 1 2}"
OUT_SUFFIX="${OUT_SUFFIX:-pure_surface_visibility_${START}_${END}_s${STRIDE}}"
STRUCTURAL_VOXEL_SIZE="${STRUCTURAL_VOXEL_SIZE:-0.10}"
VIEWER_STRIDE="${VIEWER_STRIDE:-1}"
PULL_RESULTS="${PULL_RESULTS:-0}"
SPLIT_LARGE_FINE_OBJECTS="${SPLIT_LARGE_FINE_OBJECTS:-1}"
LOCAL_GEOM_MIN_POINTS="${LOCAL_GEOM_MIN_POINTS:-2000}"

LOCAL_DRIVABILITY_PCD="${LOCAL_DRIVABILITY_PCD:-/Users/skkac/Work/SCAN/drivability_cpp/output/MT20260616-175807_drivable_points_collision_arm64_wallbfs.pcd}"
REMOTE_DRIVABILITY_PCD="${REMOTE_DRIVABILITY_PCD:-${REMOTE_WORK}/structural_priors/MT20260616-175807_drivable_points_collision_arm64_wallbfs.pcd}"

GEOMETRY_DIR="${GEOMETRY_DIR:-${REMOTE_WORK}/geometry_guidance_${OUT_SUFFIX}}"
REFINE_DIR="${REFINE_DIR:-${REMOTE_WORK}/geometry_refine_${OUT_SUFFIX}}"
PROJECTION_DIR="${PROJECTION_DIR:-${REMOTE_WORK}/priority_projection_${OUT_SUFFIX}}"
TARGET_DIR="${TARGET_DIR:-${REMOTE_WORK}/frame_targets_${OUT_SUFFIX}}"
STRUCTURAL_DIR="${STRUCTURAL_DIR:-${REMOTE_WORK}/structural_region_field_${OUT_SUFFIX}}"
ATTACHMENT_DIR="${ATTACHMENT_DIR:-${REMOTE_WORK}/frame_targets_attachment_${OUT_SUFFIX}}"
OBJECT_DIR="${OBJECT_DIR:-${REMOTE_WORK}/frame_objects_attachment_${OUT_SUFFIX}}"
VIEWER_DIR="${VIEWER_DIR:-${REMOTE_WORK}/frame_object_viewer_attachment_${OUT_SUFFIX}}"
LOCAL_GEOM_VIEWER_DIR="${LOCAL_GEOM_VIEWER_DIR:-${REMOTE_WORK}/frame_object_viewer_attachment_localgeom_${OUT_SUFFIX}}"

LOCAL_PULL_DIR="${LOCAL_PULL_DIR:-/Users/skkac/Work/SCAN/new_route/server_parking_priority_s10/${OUT_SUFFIX}}"

if [[ ! -f "${LOCAL_DRIVABILITY_PCD}" ]]; then
  echo "missing local drivability PCD: ${LOCAL_DRIVABILITY_PCD}" >&2
  exit 1
fi

echo "[sync] route scripts -> ${REMOTE_HOST}:${REMOTE_REPO}/scripts"
rsync -az \
  scripts/run_parking_safe_semantic_prior_route.sh \
  scripts/build_structural_region_field.py \
  scripts/classify_surface_attachment.py \
  scripts/fuse_targets_to_objects.py \
  scripts/export_frame_target_objects_for_viewer.py \
  scripts/build_local_geometry_split_candidates.py \
  scripts/split_priority_objects_by_local_geometry.py \
  scripts/qa_viewer_candidate.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"

echo "[sync] drivability prior -> ${REMOTE_HOST}:${REMOTE_DRIVABILITY_PCD}"
ssh "${REMOTE_HOST}" "mkdir -p '$(dirname "${REMOTE_DRIVABILITY_PCD}")'"
rsync -az --ignore-existing "${LOCAL_DRIVABILITY_PCD}" "${REMOTE_HOST}:${REMOTE_DRIVABILITY_PCD}"

remote_cmd=$(cat <<REMOTE
set -euo pipefail
cd '${REMOTE_REPO}'
source '${REMOTE_VENV}/bin/activate'
export RUN='${RUN}'
export OVERWRITE='${OVERWRITE}'
export BUILD_TARGETS=1
export BUILD_OBJECTS=0
export RUN_QA=0
export START='${START}'
export END='${END}'
export STRIDE='${STRIDE}'
export CAMS='${CAMS}'
export OUT_SUFFIX='${OUT_SUFFIX}'
export GEOMETRY_DIR='${GEOMETRY_DIR}'
export REFINE_DIR='${REFINE_DIR}'
export PROJECTION_DIR='${PROJECTION_DIR}'
export TARGET_DIR='${TARGET_DIR}'
scripts/run_parking_safe_semantic_prior_route.sh

if [[ '${RUN}' == '1' ]]; then
  rm -rf '${STRUCTURAL_DIR}' '${ATTACHMENT_DIR}' '${OBJECT_DIR}' '${VIEWER_DIR}'
  mkdir -p '${STRUCTURAL_DIR}' '${ATTACHMENT_DIR}' '${OBJECT_DIR}' '${VIEWER_DIR}'
  python scripts/build_structural_region_field.py \
    --drivability-pcd '${REMOTE_DRIVABILITY_PCD}' \
    --output-npz '${STRUCTURAL_DIR}/structural_region_field.npz' \
    --report '${STRUCTURAL_DIR}/structural_region_field_report.json' \
    --voxel-size '${STRUCTURAL_VOXEL_SIZE}'
  python scripts/classify_surface_attachment.py \
    --targets-jsonl '${TARGET_DIR}/frame_targets.jsonl' \
    --target-ply '${TARGET_DIR}/frame_targets.ply' \
    --structural-field '${STRUCTURAL_DIR}/structural_region_field.npz' \
    --output-jsonl '${ATTACHMENT_DIR}/frame_targets_attachment.jsonl' \
    --report '${ATTACHMENT_DIR}/surface_attachment_report.json'
  python scripts/fuse_targets_to_objects.py \
    --targets '${ATTACHMENT_DIR}/frame_targets_attachment.jsonl' \
    --output-dir '${OBJECT_DIR}' \
    --strict-surface-labels \
    --fallback-zone-scan \
    --write-ply
  python scripts/export_frame_target_objects_for_viewer.py \
    --targets-jsonl '${ATTACHMENT_DIR}/frame_targets_attachment.jsonl' \
    --target-ply '${TARGET_DIR}/frame_targets.ply' \
    --objects-jsonl '${OBJECT_DIR}/objects.jsonl' \
    --output-dir '${VIEWER_DIR}' \
    --stride '${VIEWER_STRIDE}' \
    --keep-target-list
  python scripts/qa_viewer_candidate.py \
    --ply '${VIEWER_DIR}/frame_object_points_stride10.ply' \
    --objects-jsonl '${VIEWER_DIR}/frame_objects_viewer.jsonl' \
    --output-json '${VIEWER_DIR}/viewer_candidate_qa.json' \
    --output-md '${VIEWER_DIR}/viewer_candidate_qa.md' || true
  if [[ '${SPLIT_LARGE_FINE_OBJECTS}' == '1' ]]; then
    rm -rf '${LOCAL_GEOM_VIEWER_DIR}'
    mkdir -p '${LOCAL_GEOM_VIEWER_DIR}'
    python scripts/build_local_geometry_split_candidates.py \
      --objects-jsonl '${VIEWER_DIR}/frame_objects_viewer.jsonl' \
      --output-jsonl '${LOCAL_GEOM_VIEWER_DIR}/local_geometry_split_candidates.jsonl' \
      --report-json '${LOCAL_GEOM_VIEWER_DIR}/local_geometry_split_candidates_report.json' \
      --labels railing,car \
      --min-points '${LOCAL_GEOM_MIN_POINTS}' \
      --require-reasons large_fine_object,large_single_target_object,railing_not_linear,railing_extent_too_large,car_extent_suspicious,car_surface_like
    python scripts/split_priority_objects_by_local_geometry.py \
      --input-ply '${VIEWER_DIR}/frame_object_points_stride10.ply' \
      --objects-jsonl '${VIEWER_DIR}/frame_objects_viewer.jsonl' \
      --conflicts-jsonl '${LOCAL_GEOM_VIEWER_DIR}/local_geometry_split_candidates.jsonl' \
      --output-dir '${LOCAL_GEOM_VIEWER_DIR}' \
      --output-prefix frame_object_points_local_geometry \
      --local-voxel-size 0.28 \
      --min-cell-points 10 \
      --min-child-points 80 \
      --min-unknown-child-points 160 \
      --railing-keep-linearity 0.78 \
      --railing-max-minor-extent 0.45 \
      --horizontal-label ground \
      --cell-connectivity 26
    cp '${LOCAL_GEOM_VIEWER_DIR}/frame_object_points_local_geometry.ply' '${LOCAL_GEOM_VIEWER_DIR}/frame_object_points_stride10.ply'
    cp '${LOCAL_GEOM_VIEWER_DIR}/frame_object_points_local_geometry.jsonl' '${LOCAL_GEOM_VIEWER_DIR}/frame_objects_viewer.jsonl'
    python scripts/qa_viewer_candidate.py \
      --ply '${LOCAL_GEOM_VIEWER_DIR}/frame_object_points_stride10.ply' \
      --objects-jsonl '${LOCAL_GEOM_VIEWER_DIR}/frame_objects_viewer.jsonl' \
      --output-json '${LOCAL_GEOM_VIEWER_DIR}/viewer_candidate_qa.json' \
      --output-md '${LOCAL_GEOM_VIEWER_DIR}/viewer_candidate_qa.md' || true
  fi
fi

echo "target_dir=${TARGET_DIR}"
echo "attachment_dir=${ATTACHMENT_DIR}"
echo "object_dir=${OBJECT_DIR}"
echo "viewer_dir=${VIEWER_DIR}"
echo "localgeom_viewer_dir=${LOCAL_GEOM_VIEWER_DIR}"
REMOTE
)

ssh "${REMOTE_HOST}" "bash -lc $(printf '%q' "${remote_cmd}")"

if [[ "${RUN}" == "1" && "${PULL_RESULTS}" == "1" ]]; then
  echo "[pull] ${REMOTE_HOST}:${STRUCTURAL_DIR},${ATTACHMENT_DIR},${OBJECT_DIR},${VIEWER_DIR} -> ${LOCAL_PULL_DIR}"
  mkdir -p "${LOCAL_PULL_DIR}"
  rsync -az "${REMOTE_HOST}:${STRUCTURAL_DIR}/" "${LOCAL_PULL_DIR}/structural/"
  rsync -az "${REMOTE_HOST}:${ATTACHMENT_DIR}/" "${LOCAL_PULL_DIR}/targets_attachment/"
  rsync -az "${REMOTE_HOST}:${OBJECT_DIR}/" "${LOCAL_PULL_DIR}/objects/"
  rsync -az "${REMOTE_HOST}:${VIEWER_DIR}/" "${LOCAL_PULL_DIR}/viewer/"
  if [[ "${SPLIT_LARGE_FINE_OBJECTS}" == "1" ]]; then
    rsync -az "${REMOTE_HOST}:${LOCAL_GEOM_VIEWER_DIR}/" "${LOCAL_PULL_DIR}/viewer_localgeom/"
  fi
fi
