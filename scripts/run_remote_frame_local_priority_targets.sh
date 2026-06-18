#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-train}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/root/epfs/work_MT20260616-175807}"
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR:-/root/epfs/datasets/MT20260616-175807}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-${REMOTE_WORKDIR}/scripts}"
PY="${PY:-/root/epfs/conda_envs/vlm_seg/bin/python}"
SESSION_NAME="${SESSION_NAME:-frame_local_priority_s10}"
WAIT="${WAIT:-0}"
CLEAN="${CLEAN:-0}"

START="${START:-0}"
END="${END:-6180}"
STRIDE="${STRIDE:-10}"
CAMS="${CAMS:-0 1 2}"
LABELS="${LABELS:-ground wall grass car railing}"

FRAME_ROOT="${FRAME_ROOT:-${REMOTE_WORKDIR}/frames_jpeg}"
PRIORITY_DIR="${PRIORITY_DIR:-${REMOTE_WORKDIR}/geometry_refine_v1_s10_full_safe}"
PRIORITY_SUFFIX="${PRIORITY_SUFFIX:-_priority_refined}"
LX="${LX:-${REMOTE_DATASET_DIR}/MANIFOLD_MT20260616-175807.lx}"
SCAN_IMAGE_DIR_REMOTE="${SCAN_IMAGE_DIR_REMOTE:-${REMOTE_DATASET_DIR}/image}"

TARGET_DIR="${TARGET_DIR:-${REMOTE_WORKDIR}/frame_targets_priority_full_s10_v1}"
OBJECT_DIR="${OBJECT_DIR:-${REMOTE_WORKDIR}/frame_objects_priority_full_s10_v1}"
VIEWER_DIR="${VIEWER_DIR:-${REMOTE_WORKDIR}/frame_object_viewer_priority_full_s10_v1}"

SURFACE_MIN_POINTS="${SURFACE_MIN_POINTS:-80}"
MIN_TARGET_POINTS="${MIN_TARGET_POINTS:-20}"
TARGET_PROGRESS_EVERY="${TARGET_PROGRESS_EVERY:-50}"
TARGET_STRIDE_PLY="${TARGET_STRIDE_PLY:-10}"

CENTROID_DISTANCE="${CENTROID_DISTANCE:-0.45}"
BBOX_DISTANCE="${BBOX_DISTANCE:-0.45}"
COLOR_DISTANCE="${COLOR_DISTANCE:-80}"
NORMAL_ANGLE="${NORMAL_ANGLE:-30}"
SURFACE_CENTROID_DISTANCE="${SURFACE_CENTROID_DISTANCE:-0.9}"
SURFACE_BBOX_DISTANCE="${SURFACE_BBOX_DISTANCE:-0.9}"
SURFACE_COLOR_DISTANCE="${SURFACE_COLOR_DISTANCE:-95}"
SURFACE_NORMAL_ANGLE="${SURFACE_NORMAL_ANGLE:-20}"

ssh_opts=(-o BatchMode=yes)
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
needed=(
  build_frame_targets_from_priority.py
  export_frame_target_objects_for_viewer.py
  fuse_targets_to_objects.py
  project_priority_masks_to_lx.py
  build_targets_from_masks.py
  project_color.py
  project_semantic.py
  make_ply_xy_preview.py
  stride_ascii_ply.py
  config.py
)

echo "[1/4] connectivity: ${SERVER}"
ssh "${ssh_opts[@]}" -o ConnectTimeout=8 "${SERVER}" "hostname; date; mkdir -p '${REMOTE_SCRIPT_DIR}'"

echo "[2/4] syncing route scripts"
tmp_tar="$(mktemp)"
COPYFILE_DISABLE=1 tar --no-xattrs -C "${SCRIPT_DIR}" -cf "${tmp_tar}" "${needed[@]}" 2>/dev/null || \
  COPYFILE_DISABLE=1 tar -C "${SCRIPT_DIR}" -cf "${tmp_tar}" "${needed[@]}"
scp "${ssh_opts[@]}" "${tmp_tar}" "${SERVER}:/tmp/frame_local_priority_scripts.tar"
rm -f "${tmp_tar}"
ssh "${ssh_opts[@]}" "${SERVER}" "tar -C '${REMOTE_SCRIPT_DIR}' -xf /tmp/frame_local_priority_scripts.tar"

read -r -d '' remote_cmd <<EOF || true
set -euo pipefail
cd '${REMOTE_WORKDIR}'
export SCAN_IMAGE_DIR='${SCAN_IMAGE_DIR_REMOTE}'
if [[ '${CLEAN}' == '1' ]]; then
  rm -rf '${TARGET_DIR}' '${OBJECT_DIR}' '${VIEWER_DIR}'
fi
mkdir -p '${TARGET_DIR}' '${OBJECT_DIR}' '${VIEWER_DIR}'
'${PY}' '${REMOTE_SCRIPT_DIR}/build_frame_targets_from_priority.py' \
  --lx '${LX}' \
  --frame-root '${FRAME_ROOT}' \
  --priority-dir '${PRIORITY_DIR}' \
  --priority-suffix '${PRIORITY_SUFFIX}' \
  --output-dir '${TARGET_DIR}' \
  --start '${START}' --end '${END}' --stride '${STRIDE}' \
  --cams ${CAMS} \
  --labels ${LABELS} \
  --surface-min-points '${SURFACE_MIN_POINTS}' \
  --min-target-points '${MIN_TARGET_POINTS}' \
  --progress-every '${TARGET_PROGRESS_EVERY}' \
  --resume | tee '${TARGET_DIR}.log'
'${PY}' '${REMOTE_SCRIPT_DIR}/fuse_targets_to_objects.py' \
  --targets '${TARGET_DIR}/frame_targets.jsonl' \
  --output-dir '${OBJECT_DIR}' \
  --centroid-distance '${CENTROID_DISTANCE}' \
  --bbox-distance '${BBOX_DISTANCE}' \
  --color-distance '${COLOR_DISTANCE}' \
  --normal-angle '${NORMAL_ANGLE}' \
  --surface-centroid-distance '${SURFACE_CENTROID_DISTANCE}' \
  --surface-bbox-distance '${SURFACE_BBOX_DISTANCE}' \
  --surface-color-distance '${SURFACE_COLOR_DISTANCE}' \
  --surface-normal-angle '${SURFACE_NORMAL_ANGLE}' | tee '${OBJECT_DIR}.log'
'${PY}' '${REMOTE_SCRIPT_DIR}/export_frame_target_objects_for_viewer.py' \
  --targets-jsonl '${TARGET_DIR}/frame_targets.jsonl' \
  --target-ply '${TARGET_DIR}/frame_targets.ply' \
  --objects-jsonl '${OBJECT_DIR}/objects.jsonl' \
  --output-dir '${VIEWER_DIR}' \
  --stride '${TARGET_STRIDE_PLY}'
'${PY}' '${REMOTE_SCRIPT_DIR}/make_ply_xy_preview.py' \
  '${VIEWER_DIR}/frame_object_points_stride10.ply' \
  --output '${VIEWER_DIR}/frame_object_points_stride10_xy.png' \
  --max-points 800000
EOF

echo "[3/4] starting tmux session: ${SESSION_NAME}"
remote_job="/tmp/${SESSION_NAME}.sh"
remote_pid="${REMOTE_WORKDIR}/logs/${SESSION_NAME}.pid"
remote_log="${REMOTE_WORKDIR}/logs/${SESSION_NAME}.log"
printf '%s\n' "${remote_cmd}" | ssh "${ssh_opts[@]}" "${SERVER}" "cat > '${remote_job}' && chmod +x '${remote_job}'"
ssh "${ssh_opts[@]}" "${SERVER}" "if command -v tmux >/dev/null 2>&1; then tmux has-session -t '${SESSION_NAME}' 2>/dev/null && tmux kill-session -t '${SESSION_NAME}' || true; tmux new-session -d -s '${SESSION_NAME}' 'bash ${remote_job}'; else mkdir -p '${REMOTE_WORKDIR}/logs'; nohup bash '${remote_job}' > '${remote_log}' 2>&1 & echo \$! > '${remote_pid}'; fi"

echo "[4/4] started"
if [[ "${WAIT}" == "1" ]]; then
  while ssh "${ssh_opts[@]}" "${SERVER}" "if command -v tmux >/dev/null 2>&1; then tmux has-session -t '${SESSION_NAME}' 2>/dev/null; else pid=\$(cat '${remote_pid}' 2>/dev/null || true); [[ -n \"\$pid\" ]] && kill -0 \"\$pid\" 2>/dev/null; fi"; do
    ssh "${ssh_opts[@]}" "${SERVER}" "if command -v tmux >/dev/null 2>&1; then tmux capture-pane -pt '${SESSION_NAME}' -S -20 || true; else tail -40 '${remote_log}' 2>/dev/null || true; fi"
    sleep 20
  done
  ssh "${ssh_opts[@]}" "${SERVER}" "cat '${TARGET_DIR}/frame_target_summary.json'; echo ---; cat '${OBJECT_DIR}/fusion_report.json'; echo ---; cat '${VIEWER_DIR}/frame_object_viewer_export_report.json'"
else
  if ssh "${ssh_opts[@]}" "${SERVER}" "command -v tmux >/dev/null 2>&1"; then
    echo "inspect: ssh ${SERVER} 'tmux attach -t ${SESSION_NAME}'"
  else
    echo "inspect: ssh ${SERVER} 'tail -f ${remote_log}'"
  fi
fi
