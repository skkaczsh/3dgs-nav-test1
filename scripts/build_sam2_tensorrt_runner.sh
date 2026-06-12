#!/usr/bin/env bash
set -euo pipefail

CXX="${CXX:-g++}"
OUT="${OUT:-/root/epfs/sam2_tensorrt/bin/sam2_trt_runner}"
SRC="${SRC:-/root/epfs/new_route_tools/sam2_trt_runner.cpp}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

mkdir -p "$(dirname "${OUT}")"

"${CXX}" -std=c++17 -O2 \
  "${SRC}" \
  -I/usr/include/x86_64-linux-gnu \
  -I"${CUDA_HOME}/include" \
  -L/usr/lib/x86_64-linux-gnu \
  -L"${CUDA_HOME}/lib64" \
  -lnvinfer -lnvinfer_plugin -lcudart \
  -o "${OUT}"

echo "${OUT}"
