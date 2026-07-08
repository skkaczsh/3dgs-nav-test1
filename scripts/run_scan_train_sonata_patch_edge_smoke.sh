#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-scan-train}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_REPO="${REMOTE_REPO:-/root/epfs/SCAN/new_route}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/epfs/venvs/sonata-lite/bin/python}"
SONATA_REPO="${SONATA_REPO:-/root/epfs/model_side_tracks/sonata}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
BASE="${BASE:-${REMOTE_WORK}/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623}"
CROP_DIR="${CROP_DIR:-${REMOTE_WORK}/pointcloud_supervised_baseline_smoke_crops_region_labels_20260708}"
REGION_INPUT="${REGION_INPUT:-${BASE}/_cpp_region_grower_input.bin}"
PATCH_LABELS="${PATCH_LABELS:-${BASE}/energy_attach_v4_contact_evidence/geo_patches_energy_attach_v4_contact_evidence_labels.bin}"
OUT_DIR="${OUT_DIR:-${REMOTE_WORK}/sonata_patch_edge_smoke_crops_20260708}"
TMUX_SESSION="${TMUX_SESSION:-scan_sonata_patch_edge_smoke}"
RUN="${RUN:-0}"

echo "host=${SSH_HOST}"
echo "crop_dir=${CROP_DIR}"
echo "out_dir=${OUT_DIR}"

if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to launch tmux session ${TMUX_SESSION}"
  exit 0
fi

rsync -az \
  "${LOCAL_REPO}/scripts/run_sonata_crop_smoke.py" \
  "${LOCAL_REPO}/scripts/pool_point_features_to_patch_features.py" \
  "${LOCAL_REPO}/scripts/build_patch_feature_edge_evidence.py" \
  "${SSH_HOST}:${REMOTE_REPO}/scripts/"

ssh "${SSH_HOST}" bash -s <<REMOTE
set -euo pipefail
test -x "${REMOTE_PYTHON}"
test -f "${CROP_DIR}/crop_export_report.json"
test -f "${REGION_INPUT}"
test -f "${PATCH_LABELS}"
mkdir -p "${OUT_DIR}"
cat > "${OUT_DIR}/run.sh" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
exec > "${OUT_DIR}/sonata_patch_edge_smoke.log" 2>&1
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
    crop_id = crop["id"]
    crop_out = out_dir / crop_id
    crop_out.mkdir(parents=True, exist_ok=True)
    ply = crop_dir / f"{crop_id}.ply"
    labels = crop_dir / f"{crop_id}_labels.bin"
    subprocess.run([
        "${REMOTE_PYTHON}",
        "scripts/run_sonata_crop_smoke.py",
        "--input", str(ply),
        "--output-dir", str(crop_out),
        "--max-points", "120000",
        "--save-feature-npz",
    ], check=True)
    feature_npz = crop_out / f"{crop_id}_sonata_features.npz"
    patch_features = crop_out / f"{crop_id}_patch_features.npz"
    edge_csv = crop_out / f"{crop_id}_sonata_edge_evidence.csv"
    subprocess.run([
        "${REMOTE_PYTHON}",
        "scripts/pool_point_features_to_patch_features.py",
        "--labels", str(labels),
        "--point-features", str(feature_npz),
        "--output", str(patch_features),
    ], check=True)
    subprocess.run([
        "${REMOTE_PYTHON}",
        "scripts/build_patch_feature_edge_evidence.py",
        "--region-input", "${REGION_INPUT}",
        "--labels", "${PATCH_LABELS}",
        "--patch-features", str(patch_features),
        "--output", str(edge_csv),
    ], check=True)
    edge_report = json.loads((Path(str(edge_csv) + ".report.json")).read_text())
    patch_report = json.loads((Path(str(patch_features) + ".report.json")).read_text())
    runs.append({
        "id": crop_id,
        "geometry_type": crop.get("geometry_type"),
        "patch_count": patch_report["patch_count"],
        "edge_evidence_count": edge_report["written_edge_count"],
        "missing_feature_edge_count": edge_report["missing_feature_edge_count"],
        "output_dir": str(crop_out),
    })
(out_dir / "sonata_patch_edge_smoke_summary.json").write_text(json.dumps({
    "schema": "sonata-patch-edge-smoke-summary/v1",
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
