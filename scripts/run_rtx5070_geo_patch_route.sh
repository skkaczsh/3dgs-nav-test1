#!/usr/bin/env bash
set -euo pipefail

# Deprecated viewer-input geometry-first post-processing route for scan-rtx5070.
#
# This route consumes a validated viewer PLY/Object JSONL pair and rebuilds
# object boundaries from GeoPatch geometry before semantic classification.
# It is not a dense production route. Use run_rtx5070_geo_patch_energy.sh for
# the current dense 0.03m voxel patch route. Default mode is dry-run; set
# RUN=1 ALLOW_VIEWER_INPUT_ROUTE=1 only when intentionally reproducing this
# legacy viewer-input experiment.

REMOTE_HOST="${REMOTE_HOST:-scan-rtx5070}"
REMOTE_REPO="${REMOTE_REPO:-/home/zsh/Work/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"

RUN="${RUN:-0}"
ALLOW_VIEWER_INPUT_ROUTE="${ALLOW_VIEWER_INPUT_ROUTE:-0}"
OUT_SUFFIX="${OUT_SUFFIX:-geo_patch_objects_window_3000_3600_v1}"
INPUT_VIEWER_DIR="${INPUT_VIEWER_DIR:-${REMOTE_WORK}/frame_object_viewer_attachment_localgeom_pure_surface_visibility_window_3000_3600}"
INPUT_PLY="${INPUT_PLY:-${INPUT_VIEWER_DIR}/frame_object_points_stride10.ply}"
STRUCTURAL_FIELD="${STRUCTURAL_FIELD:-${REMOTE_WORK}/structural_region_field_pure_surface_visibility_window_3000_3600/structural_region_field.npz}"
SCENE_PRIOR="${SCENE_PRIOR:-${REMOTE_WORK}/mimo_scene_prior_cam1_stride30_20260620/mimo_scene_prior.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REMOTE_WORK}/${OUT_SUFFIX}}"

PATCH_VOXEL_SIZE="${PATCH_VOXEL_SIZE:-0.18}"
MIN_PATCH_POINTS="${MIN_PATCH_POINTS:-120}"
POINT_STRIDE="${POINT_STRIDE:-1}"
MERGE_COMPATIBLE_PATCHES="${MERGE_COMPATIBLE_PATCHES:-1}"
ENABLE_RANSAC_PLANE_SPLIT="${ENABLE_RANSAC_PLANE_SPLIT:-0}"
ENABLE_EVIDENCE_BFS_SPLIT="${ENABLE_EVIDENCE_BFS_SPLIT:-0}"

echo "input_ply=${INPUT_PLY}"
echo "output_dir=${OUTPUT_DIR}"
echo "deprecated_viewer_input_route=1"
echo "dense_replacement=scripts/run_rtx5070_geo_patch_energy.sh"
if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  exit 0
fi
if [[ "${ALLOW_VIEWER_INPUT_ROUTE}" != "1" ]]; then
  echo "refusing to run deprecated viewer-input route; use scripts/run_rtx5070_geo_patch_energy.sh or set ALLOW_VIEWER_INPUT_ROUTE=1 for an intentional legacy reproduction" >&2
  exit 2
fi

build_extra_args=()
if [[ "${ENABLE_RANSAC_PLANE_SPLIT}" == "1" ]]; then
  build_extra_args+=(--enable-ransac-plane-split)
fi
if [[ "${ENABLE_EVIDENCE_BFS_SPLIT}" == "1" ]]; then
  build_extra_args+=(--enable-evidence-bfs-split)
fi

echo "[sync] route scripts -> ${REMOTE_HOST}:${REMOTE_REPO}/scripts"
rsync -az \
  scripts/build_geo_patches.py \
  scripts/accumulate_patch_observations.py \
  scripts/classify_geo_objects.py \
  scripts/qa_object_voxel_overlap.py \
  scripts/qa_viewer_candidate.py \
  scripts/qa_priority_geometry_conflicts.py \
  scripts/build_semantic_viewer_index.py \
  "${REMOTE_HOST}:${REMOTE_REPO}/scripts/"

remote_cmd=$(cat <<REMOTE
set -euo pipefail
cd '${REMOTE_REPO}'
echo "input_ply=${INPUT_PLY}"
echo "output_dir=${OUTPUT_DIR}"
rm -rf '${OUTPUT_DIR}'
'${REMOTE_PYTHON}' scripts/build_geo_patches.py \
  --input-ply '${INPUT_PLY}' \
  --output-dir '${OUTPUT_DIR}' \
  --structural-field '${STRUCTURAL_FIELD}' \
  --patch-voxel-size '${PATCH_VOXEL_SIZE}' \
  --min-patch-points '${MIN_PATCH_POINTS}' \
  --point-stride '${POINT_STRIDE}' \
  ${build_extra_args[*]} \
  > '${OUTPUT_DIR}.build.log'
'${REMOTE_PYTHON}' scripts/accumulate_patch_observations.py \
  --geo-patches '${OUTPUT_DIR}/geo_patches.jsonl' \
  --scene-prior '${SCENE_PRIOR}' \
  --output-jsonl '${OUTPUT_DIR}/geo_patches.evidence.jsonl' \
  --report '${OUTPUT_DIR}/geo_patch_observation_report.json' \
  > '${OUTPUT_DIR}.observe.log'
classify_args=()
if [[ '${MERGE_COMPATIBLE_PATCHES}' == '1' ]]; then
  classify_args+=(--merge-compatible-patches)
fi
'${REMOTE_PYTHON}' scripts/classify_geo_objects.py \
  --geo-patches '${OUTPUT_DIR}/geo_patches.evidence.jsonl' \
  --geo-patch-ply '${OUTPUT_DIR}/geo_patch_points.ply' \
  --output-dir '${OUTPUT_DIR}' \
  "\${classify_args[@]}" \
  > '${OUTPUT_DIR}.classify.log'
'${REMOTE_PYTHON}' scripts/qa_viewer_candidate.py \
  --ply '${OUTPUT_DIR}/frame_object_points_stride10.ply' \
  --objects-jsonl '${OUTPUT_DIR}/frame_objects_viewer.jsonl' \
  --output-json '${OUTPUT_DIR}/viewer_candidate_qa.json' \
  --output-md '${OUTPUT_DIR}/viewer_candidate_qa.md' \
  --top-n 20
'${REMOTE_PYTHON}' scripts/qa_priority_geometry_conflicts.py \
  --objects-jsonl '${OUTPUT_DIR}/frame_objects_viewer.jsonl' \
  --output-jsonl '${OUTPUT_DIR}/geometry_conflicts.jsonl' \
  --report '${OUTPUT_DIR}/geometry_conflicts_report.json'
'${REMOTE_PYTHON}' scripts/qa_object_voxel_overlap.py \
  --ply '${OUTPUT_DIR}/frame_object_points_stride10.ply' \
  --voxel-size 0.10 \
  --output-json '${OUTPUT_DIR}/voxel_overlap_report.json' \
  > '${OUTPUT_DIR}.qa_overlap.log'
'${REMOTE_PYTHON}' scripts/build_semantic_viewer_index.py \
  --artifact-root '${REMOTE_WORK}' \
  --output tools/semantic_viewer_index.json
'${REMOTE_PYTHON}' - <<'PY'
import json
out='${OUTPUT_DIR}'
report=json.load(open(f"{out}/geo_object_classification_report.json", encoding="utf-8"))
qa=json.load(open(f"{out}/geometry_conflicts_report.json", encoding="utf-8"))
overlap=json.load(open(f"{out}/voxel_overlap_report.json", encoding="utf-8"))
print(json.dumps({
  "output_dir": out,
  "object_count": report.get("object_count"),
  "label_counts": report.get("label_counts"),
  "status_counts": report.get("status_counts"),
  "geometry_findings": qa.get("finding_count"),
  "top_geometry_reasons": qa.get("top_reasons"),
  "mixed_object_voxel_ratio": overlap.get("mixed_object_voxel_ratio"),
  "mixed_semantic_voxel_ratio": overlap.get("mixed_semantic_voxel_ratio"),
}, ensure_ascii=False, indent=2))
PY
REMOTE
)

ssh "${REMOTE_HOST}" "bash -lc $(printf '%q' "${remote_cmd}")"
