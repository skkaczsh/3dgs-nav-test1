#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-/root/epfs/sam2_tensorrt}"
PYTHON_BIN="${PYTHON_BIN:-/root/epfs/conda_envs/vlm_seg/bin/python}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "${WORK_DIR}"/{onnx,engines,logs}

echo "[1/4] Export SAM2 image encoder ONNX"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" /root/epfs/new_route_scripts/export_sam2_tensorrt_onnx.py \
  --output-dir "${WORK_DIR}/onnx" \
  --export encoder \
  --device cuda \
  2>&1 | tee "${WORK_DIR}/logs/export_sam2_encoder_onnx.log"

echo "[2/4] Export SAM2 point decoder ONNX"
# The decoder traces cleanly on CPU. CUDA export currently hits an ONNX tracer
# device-mismatch path inside SAM2's decoder repeat_image branch, while the
# exported graph itself builds and runs in TensorRT.
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" /root/epfs/new_route_scripts/export_sam2_tensorrt_onnx.py \
  --output-dir "${WORK_DIR}/onnx" \
  --batch-size "${BATCH_SIZE}" \
  --export decoder \
  --device cpu \
  2>&1 | tee "${WORK_DIR}/logs/export_sam2_decoder_b${BATCH_SIZE}_onnx.log"

echo "[3/4] Build point decoder TensorRT engine"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${TRTEXEC}" \
  --onnx="${WORK_DIR}/onnx/sam2_hiera_l_point_decoder_b${BATCH_SIZE}.onnx" \
  --saveEngine="${WORK_DIR}/engines/sam2_hiera_l_point_decoder_b${BATCH_SIZE}_fp16.plan" \
  --fp16 \
  > "${WORK_DIR}/logs/trtexec_decoder_b${BATCH_SIZE}.log" 2>&1
tail -80 "${WORK_DIR}/logs/trtexec_decoder_b${BATCH_SIZE}.log"

echo "[4/4] Build image encoder TensorRT engine"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${TRTEXEC}" \
  --onnx="${WORK_DIR}/onnx/sam2_hiera_l_image_encoder.onnx" \
  --saveEngine="${WORK_DIR}/engines/sam2_hiera_l_image_encoder_fp16.plan" \
  --fp16 \
  > "${WORK_DIR}/logs/trtexec_encoder.log" 2>&1
tail -80 "${WORK_DIR}/logs/trtexec_encoder.log"

echo "SAM2 TensorRT engines:"
ls -lh "${WORK_DIR}/engines"/sam2_hiera_l_*_fp16.plan
