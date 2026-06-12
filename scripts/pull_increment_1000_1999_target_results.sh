#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-10.0.8.114}"
SSH_PORT="${SSH_PORT:-31909}"
SSH_USER="${SSH_USER:-root}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-6}"
BIND_ADDRESS="${BIND_ADDRESS:-}"

REMOTE_BASE="${REMOTE_BASE:-/root/epfs/new_route_stage1_skymask}"
REMOTE_EVAL_DIR="${REMOTE_EVAL_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_1000_1999}"
REMOTE_TARGET_DIR="${REMOTE_TARGET_DIR:-${REMOTE_BASE}/target_object_fusion_1000_1999_surface024_fine012}"
REMOTE_STATE_FILE="${REMOTE_STATE_FILE:-${REMOTE_BASE}/logs/target_object_refresh_1000_1999.state}"

LOCAL_ROOT="${LOCAL_ROOT:-/Users/skkac/Work/SCAN}"
LOCAL_TARGET_DIR="${LOCAL_TARGET_DIR:-${LOCAL_ROOT}/server_target_object_fusion_1000_1999_surface024_fine012}"
LOCAL_REPO="${LOCAL_REPO:-${LOCAL_ROOT}/new_route}"

ssh_target="${SSH_USER}@${SSH_HOST}"
ssh_opts=(-F /dev/null -o BatchMode=yes -o "ConnectTimeout=${CONNECT_TIMEOUT}" -p "${SSH_PORT}")
rsync_ssh_opts=(-F /dev/null -o BatchMode=yes -o "ConnectTimeout=${CONNECT_TIMEOUT}" -p "${SSH_PORT}")
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts=(-F /dev/null -o BatchMode=yes -o "ConnectTimeout=${CONNECT_TIMEOUT}" -o "BindAddress=${BIND_ADDRESS}" -p "${SSH_PORT}")
  rsync_ssh_opts=(-F /dev/null -o BatchMode=yes -o "ConnectTimeout=${CONNECT_TIMEOUT}" -o "BindAddress=${BIND_ADDRESS}" -p "${SSH_PORT}")
fi
printf -v rsync_ssh '%q ' ssh "${rsync_ssh_opts[@]}"
rsync_ssh="${rsync_ssh% }"

echo "[1/4] checking remote increment state: ${ssh_target}"
if [[ -n "${BIND_ADDRESS}" ]]; then
  echo "using local bind address: ${BIND_ADDRESS}"
fi
ssh "${ssh_opts[@]}" "${ssh_target}" \
  REMOTE_EVAL_DIR="${REMOTE_EVAL_DIR}" \
  REMOTE_TARGET_DIR="${REMOTE_TARGET_DIR}" \
  REMOTE_STATE_FILE="${REMOTE_STATE_FILE}" \
  'python3 - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

eval_dir = Path(os.environ["REMOTE_EVAL_DIR"])
target_dir = Path(os.environ["REMOTE_TARGET_DIR"])
state_file = Path(os.environ["REMOTE_STATE_FILE"])

paths = {
    "ply": target_dir / "objects/object_points_identity_relabel_stride10.ply",
    "objects_jsonl": target_dir / "objects/objects_identity_relabel.jsonl",
    "target_qa": target_dir / "reports/target_object_qa.json",
    "identity_report": target_dir / "reports/identity_relabel_report.json",
    "descriptions_csv": target_dir / "reports/object_identity_descriptions.csv",
}

def info(path):
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "mtime": path.stat().st_mtime if path.exists() else None,
    }

label_records = len(list((eval_dir / "images").glob("cam*_*/sam2_prompt_v3_sky_label_merge_completion/label_records.json")))
try:
    state = int(state_file.read_text().strip())
except Exception:
    state = None

print(json.dumps({
    "label_records": label_records,
    "target_refresh_state": state,
    "label_records_minus_state": label_records - state if state is not None else None,
    "files": {key: info(path) for key, path in paths.items()},
}, ensure_ascii=False, indent=2))
PY'

echo "[2/4] pulling target/object artifacts"
mkdir -p "${LOCAL_TARGET_DIR}/objects" "${LOCAL_TARGET_DIR}/reports"
rsync -av --progress -e "${rsync_ssh}" \
  "${ssh_target}:${REMOTE_TARGET_DIR}/objects/object_points_identity_relabel_stride10.ply" \
  "${LOCAL_TARGET_DIR}/objects/"
rsync -av --progress -e "${rsync_ssh}" \
  "${ssh_target}:${REMOTE_TARGET_DIR}/objects/objects_identity_relabel.jsonl" \
  "${LOCAL_TARGET_DIR}/objects/"
rsync -av --progress -e "${rsync_ssh}" \
  "${ssh_target}:${REMOTE_TARGET_DIR}/objects/fusion_report.json" \
  "${LOCAL_TARGET_DIR}/objects/" || true
rsync -av --progress -e "${rsync_ssh}" \
  "${ssh_target}:${REMOTE_TARGET_DIR}/reports/target_object_qa.json" \
  "${LOCAL_TARGET_DIR}/reports/"
rsync -av --progress -e "${rsync_ssh}" \
  "${ssh_target}:${REMOTE_TARGET_DIR}/reports/identity_relabel_report.json" \
  "${LOCAL_TARGET_DIR}/reports/"
rsync -av --progress -e "${rsync_ssh}" \
  "${ssh_target}:${REMOTE_TARGET_DIR}/reports/object_identity_descriptions.csv" \
  "${LOCAL_TARGET_DIR}/reports/" || true

echo "[3/4] refreshing local monitor/status summaries"
cd "${LOCAL_REPO}"
BIND_ADDRESS="${BIND_ADDRESS}" python3 scripts/monitor_remote_production.py || true
python3 scripts/summarize_increment_1000_1999_status.py

echo "[4/4] local artifact summary"
LOCAL_TARGET_DIR="${LOCAL_TARGET_DIR}" \
python3 - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["LOCAL_TARGET_DIR"])
qa = json.loads((base / "reports/target_object_qa.json").read_text())
identity = json.loads((base / "reports/identity_relabel_report.json").read_text())
print(json.dumps({
    "ply": str(base / "objects/object_points_identity_relabel_stride10.ply"),
    "frames": qa.get("frames"),
    "objects": qa.get("objects"),
    "identity": {
        "objects": identity.get("objects"),
        "changed": identity.get("changed"),
        "changed_ratio": identity.get("changed_ratio"),
        "new_label_counts": identity.get("new_label_counts"),
    },
}, ensure_ascii=False, indent=2))
PY

echo "done"
