#!/usr/bin/env bash
set -euo pipefail

# Refresh the local review package for the current RTX 5070Ti parking candidate.
#
# This is the local handoff command after a remote run finishes. It checks the
# remote runtime/artifacts, pulls the review-sized outputs, rebuilds the manifest,
# validates it, and prints the viewer URL.

LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
SERVER="${SERVER:-scan-rtx5070}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
REMOTE_REPO="${REMOTE_REPO:-/home/zsh/Work/SCAN/new_route}"
MANIFEST_DIR="${MANIFEST_DIR:-${LOCAL_REPO}/server_parking_priority_s10/parking_candidate_manifest_rtx5070}"
HEALTHCHECK_OUTPUT="${HEALTHCHECK_OUTPUT:-${MANIFEST_DIR}/rtx5070_runtime_check.json}"
MANIFEST_JSON="${MANIFEST_JSON:-${MANIFEST_DIR}/manifest.json}"
MANIFEST_MD="${MANIFEST_MD:-${MANIFEST_DIR}/manifest.md}"
VALIDATION_JSON="${VALIDATION_JSON:-${MANIFEST_DIR}/validation.json}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PULL="${SKIP_PULL:-0}"
PULL_QA_CROPS="${PULL_QA_CROPS:-0}"
BIND_ADDRESS="${BIND_ADDRESS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${LOCAL_REPO}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

viewer_url() {
  python3 - <<'PY' "${MANIFEST_JSON}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
    print(data.get("viewer", {}).get("url", ""))
PY
}

log "[1/5] remote runtime/artifact healthcheck"
python3 "${SCRIPT_DIR}/check_rtx5070_parking_runtime.py" \
  --host "${SERVER}" \
  --remote-repo "${REMOTE_REPO}" \
  --remote-work "${REMOTE_WORK}" \
  --output "${HEALTHCHECK_OUTPUT}"

if [[ "${SKIP_PULL}" == "1" ]]; then
  log "[2/5] skip pull requested"
else
  log "[2/5] pull review-sized artifacts"
  SERVER="${SERVER}" \
  REMOTE_WORK="${REMOTE_WORK}" \
  LOCAL_REPO="${LOCAL_REPO}" \
  DRY_RUN="${DRY_RUN}" \
  PULL_QA_CROPS="${PULL_QA_CROPS}" \
  BIND_ADDRESS="${BIND_ADDRESS}" \
    "${SCRIPT_DIR}/pull_rtx5070_parking_candidate_surface_route.sh"
fi

if [[ "${DRY_RUN}" == "1" && "${SKIP_PULL}" != "1" ]]; then
  log "[3/5] dry-run pull completed; validating current local package"
else
  log "[3/5] build local candidate manifest"
fi
python3 "${SCRIPT_DIR}/build_rtx5070_parking_candidate_manifest.py" \
  --output-json "${MANIFEST_JSON}" \
  --output-md "${MANIFEST_MD}"

log "[4/5] validate local candidate manifest"
python3 "${SCRIPT_DIR}/validate_rtx5070_parking_candidate_manifest.py" \
  --manifest "${MANIFEST_JSON}" \
  --output "${VALIDATION_JSON}"

log "[5/5] review package ready"
cat <<EOF
manifest=${MANIFEST_JSON}
validation=${VALIDATION_JSON}
viewer=$(viewer_url)
EOF
