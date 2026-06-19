#!/usr/bin/env bash
set -euo pipefail

# Prepare TensorRT on scan-rtx5070 / Ubuntu 24.04 / CUDA 13.2.
#
# Default mode is dry-run. Set APPLY=1 to perform package installation.
# TensorRT runtime/dev and Python bindings are installed through apt with a
# CUDA 13.2 pin. Pip is used only for ONNX helper packages; the NVIDIA
# tensorrt-cu13 wheel is intentionally avoided because it is multi-GB and
# redundant with the apt packages on this host.

PYTHON_BIN="${PYTHON_BIN:-/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python}"
WORK_DIR="${WORK_DIR:-/home/zsh/Work/SCAN/work_MT20260616-175807/tensorrt_smoke}"
TRT_VERSION="${TRT_VERSION:-11.0.0.114-1+cuda13.2}"
APPLY="${APPLY:-0}"
PYTHON_ONLY="${PYTHON_ONLY:-0}"
APT_GET="${APT_GET:-apt-get}"
SUDO="${SUDO:-sudo}"
PY_PACKAGES=(
  onnx
  onnxsim
  polygraphy
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
  "python3-libnvinfer=${TRT_VERSION}"
  "python3-libnvinfer-dev=${TRT_VERSION}"
  "python3-libnvinfer-lean=${TRT_VERSION}"
  "python3-libnvinfer-dispatch=${TRT_VERSION}"
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

log "[3/5] Python ONNX helper package install"
if [[ "${APPLY}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install --retries 10 --timeout 120 "${PY_PACKAGES[@]}" \
    | tee "${WORK_DIR}/logs/pip_install_tensorrt_$(date +%Y%m%d_%H%M%S).log"
else
  {
    echo "dry_run=1"
    echo "pip install is intentionally not invoked in dry-run mode."
    printf 'would_install_python_packages='
    printf '%q ' "${PY_PACKAGES[@]}"
    printf '\n'
  } | tee "${WORK_DIR}/logs/pip_dry_run_tensorrt_$(date +%Y%m%d_%H%M%S).log"
fi

log "[4/5] TensorRT apt install ${TRT_VERSION}"
if [[ "${PYTHON_ONLY}" == "1" ]]; then
  echo "python_only=1"
  echo "skip apt TensorRT C++ runtime/dev install"
elif [[ "${APPLY}" == "1" ]]; then
  "${SUDO}" "${APT_GET}" update | tee "${WORK_DIR}/logs/apt_update_tensorrt_$(date +%Y%m%d_%H%M%S).log"
  DEBIAN_FRONTEND=noninteractive "${SUDO}" "${APT_GET}" install -y "${APT_PACKAGES[@]}" \
    | tee "${WORK_DIR}/logs/apt_install_tensorrt_$(date +%Y%m%d_%H%M%S).log"
else
  "${APT_GET}" install -s "${APT_PACKAGES[@]}" \
    | tee "${WORK_DIR}/logs/apt_dry_run_tensorrt_$(date +%Y%m%d_%H%M%S).log"
fi

log "[5/5] verify"
if [[ "${PYTHON_ONLY}" == "1" ]]; then
  "$(dirname "$0")/check_rtx5070_tensorrt_readiness.py" --host local \
    --output "${WORK_DIR}/logs/python_only_tensorrt_readiness.json"
elif [[ "${APPLY}" == "1" ]]; then
  "$(dirname "$0")/verify_rtx5070_tensorrt_env.sh"
else
  echo "dry_run=1"
  echo "set APPLY=1 to install and verify"
fi
