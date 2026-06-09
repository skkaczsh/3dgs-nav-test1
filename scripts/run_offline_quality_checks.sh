#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DELIVERY_ZIP="${DELIVERY_ZIP:-/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery.zip}"
RUN_DELIVERY_CHECK="${RUN_DELIVERY_CHECK:-1}"

cd "${ROOT_DIR}"

echo "[1/4] Python compile check"
python3 -m py_compile scripts/*.py

echo "[2/4] Remote runner dependency audit"
python3 scripts/audit_runner_dependencies.py --scripts-dir scripts

echo "[3/4] Review delivery package verification"
if [[ "${RUN_DELIVERY_CHECK}" == "1" ]]; then
  python3 scripts/verify_review_delivery_manifest.py --zip-path "${DELIVERY_ZIP}"
else
  echo "skipped: RUN_DELIVERY_CHECK=${RUN_DELIVERY_CHECK}"
fi

echo "[4/4] Core offline pytest suite"
pytest -q \
  tests/test_audit_runner_dependencies.py \
  tests/test_target_object_fusion.py \
  tests/test_vlm_scene_prompt.py \
  tests/test_patch_semantic_eval_scene_prompts.py

echo "offline quality checks passed"
