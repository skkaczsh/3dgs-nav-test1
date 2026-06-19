#!/usr/bin/env bash
set -euo pipefail

# Verify TensorRT runtime/dev readiness on scan-rtx5070.

PYTHON_BIN="${PYTHON_BIN:-/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python}"
WORK_DIR="${WORK_DIR:-/home/zsh/Work/SCAN/work_MT20260616-175807/tensorrt_smoke}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${WORK_DIR}"/{onnx,engines,cpp,logs}

echo "[1/5] Python package check"
"${PYTHON_BIN}" - <<'PY'
import torch
import tensorrt as trt
import onnx
import polygraphy

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("tensorrt_python", trt.__version__)
print("onnx", onnx.__version__)
print("polygraphy", polygraphy.__version__)
PY

echo "[2/5] C++ TensorRT header/library check"
test -f /usr/include/x86_64-linux-gnu/NvInfer.h
test -f "${CUDA_HOME}/include/cuda_runtime_api.h"
ldconfig -p > "${WORK_DIR}/logs/ldconfig_snapshot.txt"
grep -q 'libnvinfer.so' "${WORK_DIR}/logs/ldconfig_snapshot.txt"
test -x "${TRTEXEC}"

cat > "${WORK_DIR}/cpp/trt_smoke.cpp" <<'CPP'
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <memory>

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

int main() {
    std::cout << "NV_TENSORRT_MAJOR=" << NV_TENSORRT_MAJOR << "\n";
    std::cout << "NV_TENSORRT_MINOR=" << NV_TENSORRT_MINOR << "\n";
    std::cout << "NV_TENSORRT_PATCH=" << NV_TENSORRT_PATCH << "\n";
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        std::cerr << "failed to create TensorRT builder" << std::endl;
        return 2;
    }
    std::cout << "builder_ok=1\n";
    return 0;
}
CPP

g++ -std=c++17 "${WORK_DIR}/cpp/trt_smoke.cpp" \
  -I/usr/include/x86_64-linux-gnu \
  -I"${CUDA_HOME}/include" \
  -L/usr/lib/x86_64-linux-gnu \
  -L"${CUDA_HOME}/lib64" \
  -lnvinfer -lnvonnxparser -lnvinfer_plugin -lcudart \
  -o "${WORK_DIR}/cpp/trt_smoke"
"${WORK_DIR}/cpp/trt_smoke"

echo "[3/5] Create tiny ONNX model"
cat > "${WORK_DIR}/onnx/make_tiny_conv_onnx.py" <<'PY'
import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 32, 32])
w = np.random.default_rng(42).standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.01
b = np.zeros((4,), dtype=np.float32)
conv = helper.make_node("Conv", ["input", "weight", "bias"], ["conv_out"], pads=[1, 1, 1, 1], strides=[1, 1])
relu = helper.make_node("Relu", ["conv_out"], ["output"])
graph = helper.make_graph(
    [conv, relu],
    "tiny_conv",
    [x],
    [y],
    [numpy_helper.from_array(w, "weight"), numpy_helper.from_array(b, "bias")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 10
onnx.checker.check_model(model)
output = os.path.join(os.environ["WORK_DIR"], "onnx", "tiny_conv.onnx")
onnx.save(model, output)
print(output)
PY
WORK_DIR="${WORK_DIR}" "${PYTHON_BIN}" "${WORK_DIR}/onnx/make_tiny_conv_onnx.py"

echo "[4/5] Build tiny TensorRT engine"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${TRTEXEC}" \
  --onnx="${WORK_DIR}/onnx/tiny_conv.onnx" \
  --saveEngine="${WORK_DIR}/engines/tiny_conv_fp16.plan" \
  --fp16 \
  --duration=1 \
  --warmUp=0 \
  --iterations=10 \
  > "${WORK_DIR}/logs/trtexec_tiny_conv.log" 2>&1
tail -40 "${WORK_DIR}/logs/trtexec_tiny_conv.log"
grep -q 'PASSED TensorRT.trtexec' "${WORK_DIR}/logs/trtexec_tiny_conv.log"

echo "[5/5] OK: RTX 5070Ti TensorRT base environment is ready"
