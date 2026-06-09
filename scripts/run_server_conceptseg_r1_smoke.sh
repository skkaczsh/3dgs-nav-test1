#!/usr/bin/env bash
set -euo pipefail

# Side-track only. This script prepares/runs a ConceptSeg-R1 smoke without
# modifying the main SAM2+Qwen semantic route.

BASE_DIR="${BASE_DIR:-/root/epfs/model_side_tracks}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/ConceptSeg-R1}"
REPO_URL="${REPO_URL:-https://github.com/NTU-AI4X/ConceptSeg-R1.git}"
MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/ConceptSeg-R1-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/example_images/outputs_scan_smoke}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RUN_SETUP="${RUN_SETUP:-0}"
RUN_INFERENCE="${RUN_INFERENCE:-0}"

mkdir -p "${BASE_DIR}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

echo "repo: ${REPO_DIR}"
git rev-parse --short HEAD || true
echo "gpu: ${CUDA_VISIBLE_DEVICES}"

if [[ "${RUN_SETUP}" == "1" ]]; then
  if [[ ! -f "${REPO_DIR}/sam3-main.zip" || ! -f "${REPO_DIR}/all_meta.json.zip" ]]; then
    echo "Missing release assets: sam3-main.zip and/or all_meta.json.zip under ${REPO_DIR}" >&2
    echo "Download them from the ConceptSeg-R1 GitHub releases before setup." >&2
    exit 2
  fi
  bash setup.sh
fi

if [[ "${RUN_INFERENCE}" != "1" ]]; then
  echo "RUN_INFERENCE=0, preparation check only."
  echo "Set RUN_SETUP=1 only after release assets are present."
  echo "Set RUN_INFERENCE=1 after MODEL_PATH points to downloaded ConceptSeg-R1 weights."
  exit 0
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing model path: ${MODEL_PATH}" >&2
  echo "Download ConceptSeg-R1-7B weights first. If a token is required, pass it as an ephemeral HF_TOKEN env var; do not write it to disk." >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES

python "${REPO_DIR}/src/eval/inference_single_example.py" \
  --model_path "${MODEL_PATH}" \
  --infer_path "${REPO_DIR}/example_images/infer.jpg" \
  --question "railing or thin metal structure" \
  --output_path "${OUTPUT_DIR}/scan_smoke_railing_or_thin_metal_structure.png"

echo "output: ${OUTPUT_DIR}/scan_smoke_railing_or_thin_metal_structure.png"
