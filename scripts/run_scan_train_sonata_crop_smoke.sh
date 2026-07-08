#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-scan-train}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_REPO="${REMOTE_REPO:-/root/epfs/SCAN/new_route}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/epfs/venvs/sonata-lite/bin/python}"
SONATA_REPO="${SONATA_REPO:-/root/epfs/model_side_tracks/sonata}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
CROP_DIR="${CROP_DIR:-${REMOTE_WORK}/pointcloud_supervised_baseline_smoke_crops_20260708}"
INPUT_PLY="${INPUT_PLY:-${CROP_DIR}/risk_70503_9366_local.ply}"
OUT_DIR="${OUT_DIR:-${REMOTE_WORK}/sonata_crop_smoke_20260708}"
TMUX_SESSION="${TMUX_SESSION:-scan_sonata_crop_smoke}"
RUN="${RUN:-0}"

echo "host=${SSH_HOST}"
echo "python=${REMOTE_PYTHON}"
echo "input=${INPUT_PLY}"
echo "out_dir=${OUT_DIR}"

if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to launch tmux session ${TMUX_SESSION}"
  exit 0
fi

rsync -az \
  "${LOCAL_REPO}/scripts/run_sonata_crop_smoke.py" \
  "${SSH_HOST}:${REMOTE_REPO}/scripts/"

ssh "${SSH_HOST}" bash -s <<REMOTE
set -euo pipefail
test -x "${REMOTE_PYTHON}"
test -f "${INPUT_PLY}"
mkdir -p "${OUT_DIR}"
cat > "${OUT_DIR}/run.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
exec > "${OUT_DIR}/sonata_crop_smoke.log" 2>&1
cd "${REMOTE_REPO}"
export PYTHONPATH="${SONATA_REPO}:\${PYTHONPATH:-}"
"${REMOTE_PYTHON}" scripts/run_sonata_crop_smoke.py \
  --input "${INPUT_PLY}" \
  --output-dir "${OUT_DIR}" \
  --max-points 120000
date -Is > "${OUT_DIR}/DONE"
SCRIPT
chmod +x "${OUT_DIR}/run.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${OUT_DIR}/run.sh"
tmux ls
REMOTE
