#!/usr/bin/env bash
set -euo pipefail

# Prepare TensorRT on scan-rtx5070 / Ubuntu 24.04 / CUDA 13.2.
#
# Default mode is dry-run. Set APPLY=1 to perform package installation.

PYTHON_BIN="${PYTHON_BIN:-/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python}"
WORK_DIR="${WORK_DIR:-/home/zsh/Work/SCAN/work_MT20260616-175807/tensorrt_smoke}"
TRT_VERSION="${TRT_VERSION:-11.0.0.114-1+cuda13.2}"
PY_TRT_VERSION="${PY_TRT_VERSION:-11.0.0.114}"
APPLY="${APPLY:-0}"
APT_GET="${APT_GET:-apt-get}"
SUDO="${SUDO:-sudo}"
PIP_INDEX_ARGS=(
  --extra-index-url https://pypi.nvidia.com
)
PY_PACKAGES=(
  "tensorrt-cu13==${PY_TRT_VERSION}"
  onnx
  onnxsim
  polygraphy
  cuda-python
)
APT_PACKAGES=(
  "libnvinfer-dev=${TRT_VERSION}"
  "libnvinfer-headers-dev=${TRT_VERSION}"
  "libnvinfer-safe-headers-dev=${TRT_VERSION}"
  "libnvinfer-lean-dev=${TRT_VERSION}"
  "libnvinfer-dispatch-dev=${TRT_VERSION}"
  "libnvinfer-plugin-dev=${TRT_VERSION}"
  "libnvinfer-headers-plugin-dev=${TRT_VERSION}"
  "libnvinfer-vc-plugin-dev=${TRT_VERSION}"
  "libnvonnxparsers-dev=${TRT_VERSION}"
  "libnvinfer11=${TRT_VERSION}"
  "libnvinfer-lean11=${TRT_VERSION}"
  "libnvinfer-dispatch11=${TRT_VERSION}"
  "libnvinfer-plugin11=${TRT_VERSION}"
  "libnvinfer-vc-plugin11=${TRT_VERSION}"
  "libnvonnxparsers11=${TRT_VERSION}"
  "libnvinfer-bin=${TRT_VERSION}"
)

mkdir -p "${WORK_DIR}/logs"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_or_print() {
  if [[ "${APPLY}" == "1" ]]; then
    log "run: $*"
    "$@"
  else
    printf 'dry_run_cmd='
    printf '%q ' "$@"
    printf '\n'
  fi
}

log "[1/5] environment summary"
cat /etc/os-release | grep -E '^(PRETTY_NAME|VERSION_CODENAME)=' || true
readlink -f /usr/local/cuda || true
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY

log "[2/5] apt candidate check"
for pkg in libnvinfer-dev libnvinfer-bin libnvonnxparsers-dev tensorrt; do
  apt-cache policy "${pkg}" | sed -n '1,8p'
done | tee "${WORK_DIR}/logs/apt_policy_tensorrt_$(date +%Y%m%d_%H%M%S).log"

log "[3/5] Python package install ${PY_TRT_VERSION}"
if [[ "${APPLY}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install "${PIP_INDEX_ARGS[@]}" "${PY_PACKAGES[@]}" \
    | tee "${WORK_DIR}/logs/pip_install_tensorrt_$(date +%Y%m%d_%H%M%S).log"
else
  {
    echo "dry_run=1"
    echo "pip install is intentionally not invoked in dry-run mode because NVIDIA TensorRT wheels are multi-GB."
    "${PYTHON_BIN}" -m pip index versions tensorrt-cu13 --index-url https://pypi.nvidia.com | head -20 || true
    "${PYTHON_BIN}" -m pip index versions tensorrt --extra-index-url https://pypi.nvidia.com | head -20 || true
    printf 'would_install_python_packages='
    printf '%q ' "${PY_PACKAGES[@]}"
    printf '\n'
  } | tee "${WORK_DIR}/logs/pip_dry_run_tensorrt_$(date +%Y%m%d_%H%M%S).log"
fi

log "[4/5] TensorRT apt install ${TRT_VERSION}"
if [[ "${APPLY}" == "1" ]]; then
  "${SUDO}" "${APT_GET}" update | tee "${WORK_DIR}/logs/apt_update_tensorrt_$(date +%Y%m%d_%H%M%S).log"
  DEBIAN_FRONTEND=noninteractive "${SUDO}" "${APT_GET}" install -y "${APT_PACKAGES[@]}" \
    | tee "${WORK_DIR}/logs/apt_install_tensorrt_$(date +%Y%m%d_%H%M%S).log"
else
  "${APT_GET}" install -s "${APT_PACKAGES[@]}" \
    | tee "${WORK_DIR}/logs/apt_dry_run_tensorrt_$(date +%Y%m%d_%H%M%S).log"
fi

log "[5/5] verify"
if [[ "${APPLY}" == "1" ]]; then
  "$(dirname "$0")/verify_rtx5070_tensorrt_env.sh"
else
  echo "dry_run=1"
  echo "set APPLY=1 to install and verify"
fi
