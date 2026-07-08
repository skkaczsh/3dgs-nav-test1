#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-scan-train}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_REPO="${REMOTE_REPO:-/root/epfs/SCAN/new_route}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/epfs/venvs/sonata-lite/bin/python}"
SONATA_REPO="${SONATA_REPO:-/root/epfs/model_side_tracks/sonata}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
CROP_DIR="${CROP_DIR:-${REMOTE_WORK}/pointcloud_supervised_baseline_smoke_crops_20260708}"
OUT_DIR="${OUT_DIR:-${REMOTE_WORK}/sonata_smoke_crops_20260708}"
TMUX_SESSION="${TMUX_SESSION:-scan_sonata_smoke_crops}"
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
  "${LOCAL_REPO}/scripts/run_sonata_crop_smoke.py" \
  "${SSH_HOST}:${REMOTE_REPO}/scripts/"

ssh "${SSH_HOST}" bash -s <<REMOTE
set -euo pipefail
test -x "${REMOTE_PYTHON}"
test -f "${CROP_DIR}/crop_export_report.json"
mkdir -p "${OUT_DIR}"
cat > "${OUT_DIR}/run.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
exec > "${OUT_DIR}/sonata_smoke_crops.log" 2>&1
cd "${REMOTE_REPO}"
export PYTHONPATH="${SONATA_REPO}:\${PYTHONPATH:-}"
"${REMOTE_PYTHON}" - <<'PY'
import json
import subprocess
from pathlib import Path

crop_dir = Path("${CROP_DIR}")
out_dir = Path("${OUT_DIR}")
report = json.loads((crop_dir / "crop_export_report.json").read_text())
runs = []
for crop in report["crops"]:
    name = Path(crop["output_ply"]).name
    crop_out = out_dir / crop["id"]
    crop_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "${REMOTE_PYTHON}",
        "scripts/run_sonata_crop_smoke.py",
        "--input",
        str(crop_dir / name),
        "--output-dir",
        str(crop_out),
        "--max-points",
        "120000",
    ]
    print("RUN", crop["id"], flush=True)
    subprocess.run(cmd, check=True)
    runs.append({"id": crop["id"], "geometry_type": crop.get("geometry_type"), "output_dir": str(crop_out)})
(out_dir / "sonata_smoke_crops_summary.json").write_text(json.dumps({
    "schema": "sonata-smoke-crops-summary/v1",
    "crop_count": len(runs),
    "runs": runs,
}, indent=2), encoding="utf-8")
PY
date -Is > "${OUT_DIR}/DONE"
SCRIPT
chmod +x "${OUT_DIR}/run.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${OUT_DIR}/run.sh"
tmux ls
REMOTE
