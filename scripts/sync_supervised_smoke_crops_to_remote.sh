#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-scan-train}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/root/epfs/SCAN/work_MT20260616-175807}"
LOCAL_DIR="${LOCAL_DIR:-${LOCAL_REPO}/server_parking_priority_s10/pointcloud_supervised_baseline_smoke_crops_20260708}"
REMOTE_DIR="${REMOTE_DIR:-${REMOTE_WORK}/pointcloud_supervised_baseline_smoke_crops_20260708}"
DRY_RUN="${DRY_RUN:-0}"

REPORT="${LOCAL_DIR}/crop_export_report.json"

echo "host=${SSH_HOST}"
echo "local_dir=${LOCAL_DIR}"
echo "remote_dir=${REMOTE_DIR}"

test -d "${LOCAL_DIR}"
test -f "${REPORT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run=1"
  echo "ssh ${SSH_HOST} mkdir -p ${REMOTE_DIR}"
  echo "rsync -az --progress ${LOCAL_DIR}/ ${SSH_HOST}:${REMOTE_DIR}/"
  echo "ssh ${SSH_HOST} python3 - ${REMOTE_DIR} < remote_hash_verify.py"
  exit 0
fi

ssh "${SSH_HOST}" mkdir -p "${REMOTE_DIR}"
rsync -az --progress "${LOCAL_DIR}/" "${SSH_HOST}:${REMOTE_DIR}/"

ssh "${SSH_HOST}" python3 - "${REMOTE_DIR}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

remote_dir = Path(sys.argv[1])
report_path = remote_dir / "crop_export_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
errors: list[str] = []

for crop in report.get("crops", []):
    source_name = Path(str(crop.get("output_ply", ""))).name
    ply_path = remote_dir / source_name
    if not ply_path.is_file():
        errors.append(f"missing={source_name}")
        continue
    digest = hashlib.sha256()
    with ply_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = str(crop.get("sha256", ""))
    if actual != expected:
        errors.append(f"sha256_mismatch={source_name}:expected={expected}:actual={actual}")

result = {
    "remote_dir": str(remote_dir),
    "crop_count": len(report.get("crops", [])),
    "passed": not errors,
    "errors": errors,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit(1)
PY
