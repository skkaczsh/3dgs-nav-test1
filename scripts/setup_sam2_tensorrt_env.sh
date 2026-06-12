#!/usr/bin/env bash
set -euo pipefail

# Install the SAM2 TensorRT toolchain on scan-train.
#
# This deliberately keeps the current Python SAM2 pipeline usable:
# - Python export/prototyping packages are installed into vlm_seg.
# - C++ TensorRT headers/libs are installed from NVIDIA apt packages.
# - TensorRT C++ is pinned to CUDA 11.8 to match /usr/local/cuda on scan-train.

PYTHON_BIN="${PYTHON_BIN:-/root/epfs/conda_envs/vlm_seg/bin/python}"
WORK_DIR="${WORK_DIR:-/root/epfs/sam2_tensorrt}"
CUDA_REPO_LIST="${CUDA_REPO_LIST:-/etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list}"
CUDA_REPO_URL="${CUDA_REPO_URL:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/}"
TRT_VERSION="${TRT_VERSION:-10.9.0.34-1+cuda11.8}"

mkdir -p "${WORK_DIR}"/{onnx,engines,cpp,logs,third_party,build}

echo "[1/4] Install Python ONNX/TensorRT helpers into ${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  onnx onnxsim polygraphy tensorrt-cu12 cuda-python \
  | tee "${WORK_DIR}/logs/pip_install_$(date +%Y%m%d_%H%M%S).log"

echo "[2/4] Configure NVIDIA CUDA apt repository"
cd /tmp
wget -q -O cuda-keyring_1.1-1_all.deb \
  "${CUDA_REPO_URL}/cuda-keyring_1.1-1_all.deb"
dpkg -i cuda-keyring_1.1-1_all.deb
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] ${CUDA_REPO_URL} /" \
  > "${CUDA_REPO_LIST}"
apt-get update | tee "${WORK_DIR}/logs/apt_update_$(date +%Y%m%d_%H%M%S).log"

echo "[3/4] Install TensorRT C++ dev/runtime ${TRT_VERSION}"
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  "libnvinfer-dev=${TRT_VERSION}" \
  "libnvinfer-headers-dev=${TRT_VERSION}" \
  "libnvinfer-lean-dev=${TRT_VERSION}" \
  "libnvinfer-dispatch-dev=${TRT_VERSION}" \
  "libnvinfer-plugin-dev=${TRT_VERSION}" \
  "libnvinfer-headers-plugin-dev=${TRT_VERSION}" \
  "libnvinfer-vc-plugin-dev=${TRT_VERSION}" \
  "libnvonnxparsers-dev=${TRT_VERSION}" \
  "libnvinfer10=${TRT_VERSION}" \
  "libnvinfer-lean10=${TRT_VERSION}" \
  "libnvinfer-dispatch10=${TRT_VERSION}" \
  "libnvinfer-plugin10=${TRT_VERSION}" \
  "libnvinfer-vc-plugin10=${TRT_VERSION}" \
  "libnvonnxparsers10=${TRT_VERSION}" \
  "libnvinfer-bin=${TRT_VERSION}" \
  | tee "${WORK_DIR}/logs/apt_install_tensorrt_$(date +%Y%m%d_%H%M%S).log"

echo "[4/4] Verify install"
"$(dirname "$0")/verify_sam2_tensorrt_env.sh"

