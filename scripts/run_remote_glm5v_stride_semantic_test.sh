#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-train}"
SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-}"
SSH_USER="${SSH_USER:-root}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"
TAR_BIN="${TAR_BIN:-bsdtar}"

START_FRAME="${START_FRAME:-2000}"
END_FRAME="${END_FRAME:-2999}"
FRAME_STRIDE="${FRAME_STRIDE:-10}"
CAMERAS="${CAMERAS:-}"
SHARDS="${SHARDS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
VLM_TIMEOUT="${VLM_TIMEOUT:-60}"
VLM_IMAGE_MAX_SIZE="${VLM_IMAGE_MAX_SIZE:-512}"
VLM_DISABLE_THINKING="${VLM_DISABLE_THINKING:-1}"
VLM_RETRIES="${VLM_RETRIES:-2}"
VLM_RETRY_SLEEP="${VLM_RETRY_SLEEP:-5}"
VLM_ENDPOINT="${VLM_ENDPOINT:-https://ai.1cc.ai/v1/chat/completions}"
VLM_MODEL="${VLM_MODEL:-glm-5v-turbo}"
VLM_RUN_SLUG="${VLM_RUN_SLUG:-glm5v}"
BASE_MANIFEST="${BASE_MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_2000_2999.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_glm5v_stride_2000_2999_s10}"
SAM_SOURCE_DIRS="${SAM_SOURCE_DIRS:-/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_rle50}"
SAM_MASKS_DIR="${SAM_MASKS_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_rle50_linked}"
REQUIRE_SAM_JSON="${REQUIRE_SAM_JSON:-1}"
TARGET_OUTPUT_DIR="${TARGET_OUTPUT_DIR:-/root/epfs/new_route_stage1_skymask/target_object_fusion_glm5v_2000_2999_s10_geometry_guard_strict}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-/Users/skkac/Work/SCAN/server_glm5v_2000_2999_s10_geometry_guard_strict}"
SEMANTIC_ROOT="${SEMANTIC_ROOT:-/root/epfs/manifold_3dgs_project/semantic_eval}"

if [[ -z "${VLM_API_KEY:-}" ]]; then
  echo "VLM_API_KEY is required in the local environment; it is forwarded but not written to disk." >&2
  exit 2
fi

server_target="${SERVER}"
scp_target="${SERVER}"
ssh_opts=()
scp_opts=()
if [[ -n "${SSH_HOST}" ]]; then
  ssh_opts+=("-F" "/dev/null")
  scp_opts+=("-F" "/dev/null")
  if [[ -n "${SSH_PORT}" ]]; then
    ssh_opts+=("-p" "${SSH_PORT}")
    scp_opts+=("-P" "${SSH_PORT}")
  fi
  server_target="${SSH_USER}@${SSH_HOST}"
  scp_target="${SSH_USER}@${SSH_HOST}"
fi
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
  scp_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
fi

echo "[1/6] connectivity: ${server_target}"
ssh "${ssh_opts[@]}" -o ConnectTimeout=8 "${server_target}" 'hostname; date'

echo "[2/6] sync scripts"
COPYFILE_DISABLE=1 "${TAR_BIN}" --no-xattrs -C "${LOCAL_SCRIPT_DIR}" --exclude='__pycache__' --exclude='._*' -cf - . \
  | ssh "${ssh_opts[@]}" "${server_target}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf -"

echo "[3/6] run VLM stride semantic test in tmux"
REMOTE_KEY_FIFO="/tmp/glm5v_stride_semantic_key.$$"
REMOTE_STATUS_FILE="${OUTPUT_DIR}/_sharded_work/logs/remote_run_status.json"
remote_cmd=$(cat <<REMOTE
set -euo pipefail
mkdir -p '${OUTPUT_DIR}/_sharded_work/logs'
trap 'code=\$?; printf "{\"exit_code\":%s,\"finished_at\":\"%s\"}\\n" "\${code}" "\$(date -Is)" > "${REMOTE_STATUS_FILE}"' EXIT
cd '${REMOTE_SCRIPT_DIR}'
IFS= read -r VLM_API_KEY < '${REMOTE_KEY_FIFO}'
rm -f '${REMOTE_KEY_FIFO}'
export VLM_API_KEY
camera_args=()
if [[ -n '${CAMERAS}' ]]; then
  IFS=',' read -r -a camera_list <<< '${CAMERAS}'
  for camera in "\${camera_list[@]}"; do
    if [[ -n "\${camera}" ]]; then
      camera_args+=(--camera "\${camera}")
    fi
  done
fi
python3 patch_semantic_eval_vlm_auth.py --semantic-root '${SEMANTIC_ROOT}' --report '${OUTPUT_DIR}/_sharded_work/logs/vlm_auth_patch_report.json'
python3 filter_manifest_frame_stride.py --input '${BASE_MANIFEST}' --output '${OUTPUT_DIR}/manifest_${START_FRAME}_${END_FRAME}_s${FRAME_STRIDE}.json' --start-frame '${START_FRAME}' --end-frame '${END_FRAME}' --frame-stride '${FRAME_STRIDE}' "\${camera_args[@]}"
MANIFEST='${OUTPUT_DIR}/manifest_${START_FRAME}_${END_FRAME}_s${FRAME_STRIDE}.json' \
OUTPUT_DIR='${OUTPUT_DIR}' \
SAM_SOURCE_DIRS='${SAM_SOURCE_DIRS}' \
SAM_MASKS_DIR='${SAM_MASKS_DIR}' \
REQUIRE_SAM_JSON='${REQUIRE_SAM_JSON}' \
START_INDEX=0 \
END_INDEX=999999 \
SHARDS='${SHARDS}' \
CHUNK_SIZE='${CHUNK_SIZE}' \
MAX_TOKENS='${MAX_TOKENS}' \
VLM_IMAGE_MAX_SIZE='${VLM_IMAGE_MAX_SIZE}' \
VLM_DISABLE_THINKING='${VLM_DISABLE_THINKING}' \
VLM_RETRIES='${VLM_RETRIES}' \
VLM_RETRY_SLEEP='${VLM_RETRY_SLEEP}' \
VLM_TIMEOUT='${VLM_TIMEOUT}' \
PATCH_SCENE_PROMPTS=1 \
VLM_ENDPOINT='${VLM_ENDPOINT}' \
VLM_MODEL='${VLM_MODEL}' \
bash ./run_server_semantic_completion_sharded.sh
python3 build_targets_from_masks.py --semantic-eval-dir '${OUTPUT_DIR}' --combo sam2_prompt_v3_sky_label_merge_completion --color-dir /root/epfs/new_route_stage1_skymask/output --output-dir '${TARGET_OUTPUT_DIR}/raw_targets' --frames-from-semantic-dir --start '${START_FRAME}' --end '${END_FRAME}' --resume --write-ply
python3 geometry_guard_targets.py --input-targets '${TARGET_OUTPUT_DIR}/raw_targets/targets' --output-targets '${TARGET_OUTPUT_DIR}/targets' --report '${TARGET_OUTPUT_DIR}/reports/geometry_guard_report.json'
python3 fuse_targets_to_objects.py --targets '${TARGET_OUTPUT_DIR}/targets' --output-dir '${TARGET_OUTPUT_DIR}/objects' --min-merge-confidence 0.5 --strict-surface-labels --write-ply
python3 qa_target_object_fusion.py --target-report '${TARGET_OUTPUT_DIR}/raw_targets/reports/target_build_report.json' --objects-jsonl '${TARGET_OUTPUT_DIR}/objects/objects.jsonl' --fusion-report '${TARGET_OUTPUT_DIR}/objects/fusion_report.json' --zones-json '${TARGET_OUTPUT_DIR}/objects/zones.json' --output '${TARGET_OUTPUT_DIR}/reports/target_object_qa.json'
python3 stride_ascii_ply.py '${TARGET_OUTPUT_DIR}/objects/object_centroids.ply' '${TARGET_OUTPUT_DIR}/objects/object_points_${VLM_RUN_SLUG}_geometry_guard_strict_stride10.ply' --stride 10
python3 make_ply_xy_preview.py '${TARGET_OUTPUT_DIR}/objects/object_points_${VLM_RUN_SLUG}_geometry_guard_strict_stride10.ply' --output '${TARGET_OUTPUT_DIR}/objects/object_points_${VLM_RUN_SLUG}_geometry_guard_strict_stride10_xy.png' --max-points 800000
REMOTE
)
ssh "${ssh_opts[@]}" "${server_target}" "rm -f '${REMOTE_KEY_FIFO}'; mkfifo -m 600 '${REMOTE_KEY_FIFO}'; tmux kill-session -t glm5v_stride_semantic 2>/dev/null || true; tmux new-session -d -s glm5v_stride_semantic $(printf '%q' "bash -lc $(printf '%q' "${remote_cmd}")")"
printf '%s\n' "${VLM_API_KEY}" | ssh "${ssh_opts[@]}" "${server_target}" "cat > '${REMOTE_KEY_FIFO}'"

echo "[4/6] wait for tmux completion"
while ssh "${ssh_opts[@]}" "${server_target}" "tmux has-session -t glm5v_stride_semantic 2>/dev/null"; do
  ssh "${ssh_opts[@]}" "${server_target}" "find '${OUTPUT_DIR}/_sharded_work/logs' '${TARGET_OUTPUT_DIR}/reports' -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -8" || true
  sleep 30
done
remote_status="$(ssh "${ssh_opts[@]}" "${server_target}" "cat '${REMOTE_STATUS_FILE}' 2>/dev/null || true")"
if [[ -z "${remote_status}" || "${remote_status}" != *'"exit_code":0'* ]]; then
  echo "remote run failed or did not write a success status: ${remote_status:-<missing>}" >&2
  ssh "${ssh_opts[@]}" "${server_target}" "find '${OUTPUT_DIR}/_sharded_work/logs' -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -20" >&2 || true
  exit 1
fi

echo "[5/6] pull reports and stride PLY"
mkdir -p "${LOCAL_OUTPUT_DIR}/reports" "${LOCAL_OUTPUT_DIR}/objects" "${LOCAL_OUTPUT_DIR}/semantic_logs"
scp "${scp_opts[@]}" "${scp_target}:${TARGET_OUTPUT_DIR}/reports/geometry_guard_report.json" "${LOCAL_OUTPUT_DIR}/reports/" || true
scp "${scp_opts[@]}" "${scp_target}:${TARGET_OUTPUT_DIR}/reports/target_object_qa.json" "${LOCAL_OUTPUT_DIR}/reports/" || true
scp "${scp_opts[@]}" "${scp_target}:${TARGET_OUTPUT_DIR}/objects/fusion_report.json" "${LOCAL_OUTPUT_DIR}/objects/" || true
scp "${scp_opts[@]}" "${scp_target}:${TARGET_OUTPUT_DIR}/objects/object_points_${VLM_RUN_SLUG}_geometry_guard_strict_stride10.ply" "${LOCAL_OUTPUT_DIR}/objects/" || true
scp "${scp_opts[@]}" "${scp_target}:${TARGET_OUTPUT_DIR}/objects/object_points_${VLM_RUN_SLUG}_geometry_guard_strict_stride10_xy.png" "${LOCAL_OUTPUT_DIR}/objects/" || true
scp "${scp_opts[@]}" "${scp_target}:${OUTPUT_DIR}/_sharded_work/logs/label_records_report.json" "${LOCAL_OUTPUT_DIR}/semantic_logs/" || true

echo "[6/6] done: ${LOCAL_OUTPUT_DIR}"
