#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DELIVERY_ZIP="${DELIVERY_ZIP:-/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery.zip}"
RUN_DELIVERY_CHECK="${RUN_DELIVERY_CHECK:-1}"
OFFLINE_QA_REPORT="${OFFLINE_QA_REPORT:-/Users/skkac/Work/SCAN/route_status_20260610/offline_quality_latest.json}"

cd "${ROOT_DIR}"

echo "[1/6] Python compile check"
python3 -m py_compile scripts/*.py

echo "[2/6] Sensitive token scan"
python3 scripts/scan_sensitive_tokens.py --root .

echo "[3/6] Remote runner dependency audit"
python3 scripts/audit_runner_dependencies.py --scripts-dir scripts

echo "[4/6] Review delivery package verification"
if [[ "${RUN_DELIVERY_CHECK}" == "1" ]]; then
  python3 scripts/verify_review_delivery_manifest.py --zip-path "${DELIVERY_ZIP}"
else
  echo "skipped: RUN_DELIVERY_CHECK=${RUN_DELIVERY_CHECK}"
fi

echo "[5/6] Server resume command plan validation"
python3 scripts/prepare_server_resume_commands.py
python3 scripts/validate_server_resume_commands.py \
  --report /Users/skkac/Work/SCAN/route_status_20260610/server_resume_commands_validation.json
python3 scripts/validate_server_resume_outputs.py \
  --output /Users/skkac/Work/SCAN/route_status_20260610/server_resume_output_validation.json

echo "[6/6] Core offline pytest suite"
pytest -q \
  tests/test_audit_runner_dependencies.py \
  tests/test_offline_quality_runner.py \
  tests/test_prepare_server_resume_commands.py \
  tests/test_prepare_server_resume_report.py \
  tests/test_route_status_summary.py \
  tests/test_scan_sensitive_tokens.py \
  tests/test_target_object_fusion.py \
  tests/test_validate_server_resume_commands.py \
  tests/test_validate_server_resume_outputs.py \
  tests/test_vlm_scene_prompt.py \
  tests/test_patch_semantic_eval_scene_prompts.py

python3 - <<'PY' "${OFFLINE_QA_REPORT}" "${DELIVERY_ZIP}" "${RUN_DELIVERY_CHECK}"
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

report_path = Path(sys.argv[1])
delivery_zip = sys.argv[2]
run_delivery_check = sys.argv[3]

try:
    git_head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    git_head = ""

report = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "passed": True,
    "git_head": git_head,
    "delivery_zip": delivery_zip,
    "run_delivery_check": run_delivery_check == "1",
    "checks": [
        "python_compile",
        "sensitive_token_scan",
        "remote_runner_dependency_audit",
        "review_delivery_package_verification" if run_delivery_check == "1" else "review_delivery_package_verification_skipped",
        "server_resume_command_plan_validation",
        "server_resume_output_validation_non_strict",
        "core_offline_pytest",
    ],
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"offline QA report: {report_path}")
PY

echo "offline quality checks passed"
