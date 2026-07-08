#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-scan-train}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_REPO="${REMOTE_REPO:-/root/epfs/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/opt/conda/envs/depth-anything-3/bin/python}"
CROP_DIR="${CROP_DIR:-${REMOTE_WORK}/pointcloud_supervised_baseline_smoke_crops_20260708}"
OUT_DIR="${OUT_DIR:-${REMOTE_WORK}/pointcloud_supervised_baseline_smoke_probe_20260708}"
TMUX_SESSION="${TMUX_SESSION:-scan_supervised_smoke_probe}"
RUN="${RUN:-0}"

echo "host=${SSH_HOST}"
echo "python=${REMOTE_PYTHON}"
echo "crop_dir=${CROP_DIR}"
echo "out_dir=${OUT_DIR}"

if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to launch tmux session ${TMUX_SESSION}"
  exit 0
fi

rsync -az \
  "${LOCAL_REPO}/scripts/probe_supervised_smoke_crops.py" \
  "${SSH_HOST}:${REMOTE_REPO}/scripts/"

ssh "${SSH_HOST}" bash -s <<REMOTE
set -euo pipefail
test -x "${REMOTE_PYTHON}"
test -f "${CROP_DIR}/crop_export_report.json"
mkdir -p "${OUT_DIR}"
cat > "${OUT_DIR}/run.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd "${REMOTE_REPO}"
"${REMOTE_PYTHON}" scripts/probe_supervised_smoke_crops.py \
  --crop-dir "${CROP_DIR}" \
  --output "${OUT_DIR}/feature_probe_report.json" \
  > "${OUT_DIR}/feature_probe.log" 2>&1
date -Is > "${OUT_DIR}/DONE"
SCRIPT
chmod +x "${OUT_DIR}/run.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${OUT_DIR}/run.sh"
tmux ls
REMOTE
