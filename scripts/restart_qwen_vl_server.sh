#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
MODEL_DIR="${MODEL_DIR:-/root/epfs/models/Qwen3.6-35B-A3B-Q4_K_M}"
MODEL="${MODEL:-${MODEL_DIR}/Qwen3.6-35B-A3B-Q4_K_M-00001-of-00002.gguf}"
MMPROJ="${MMPROJ:-${MODEL_DIR}/mmproj-Qwen3.6-A3B-Q4_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-/root/llama-cpp-turboquant/build/bin/llama-server}"
LOG="${LOG:-/root/epfs/qwen_vl_server_${PORT}.log}"
CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-2}"
NGL="${NGL:-99}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "missing llama-server: ${LLAMA_SERVER}" >&2
  exit 2
fi
if [[ ! -f "${MODEL}" || ! -f "${MMPROJ}" ]]; then
  echo "missing model or mmproj" >&2
  echo "MODEL=${MODEL}" >&2
  echo "MMPROJ=${MMPROJ}" >&2
  exit 2
fi

old_pids="$(pgrep -f "${LLAMA_SERVER}.*--port ${PORT}" || true)"
if [[ -n "${old_pids}" ]]; then
  echo "stopping existing llama-server on port ${PORT}: ${old_pids}"
  kill ${old_pids} || true
  sleep 5
fi

echo "starting Qwen VL server port=${PORT} ctx=${CTX_SIZE} parallel=${PARALLEL}"
nohup "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --mmproj "${MMPROJ}" \
  -ngl "${NGL}" \
  -c "${CTX_SIZE}" \
  -fa auto \
  -np "${PARALLEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  > "${LOG}" 2>&1 &
pid=$!
echo "pid=${pid} log=${LOG}"

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ready"
    curl -fsS "http://127.0.0.1:${PORT}/slots" 2>/dev/null | head -c 1000 || true
    echo
    exit 0
  fi
  sleep 2
done

echo "server did not become ready; tail log:" >&2
tail -80 "${LOG}" >&2 || true
exit 1
