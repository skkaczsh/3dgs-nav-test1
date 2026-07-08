#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-scan-train}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/opt/conda/envs/depth-anything-3/bin/python}"
SONATA_REPO="${SONATA_REPO:-/root/epfs/model_side_tracks/sonata}"
POINTCEPT_REPO="${POINTCEPT_REPO:-/root/epfs/model_side_tracks/pointcept}"

ssh "${SSH_HOST}" bash -s -- "${REMOTE_PYTHON}" "${SONATA_REPO}" "${POINTCEPT_REPO}" <<'REMOTE'
set -euo pipefail
PYTHON="$1"
SONATA_REPO="$2"
POINTCEPT_REPO="$3"

"${PYTHON}" - <<'PY'
import importlib.util
import json
mods = ["torch", "numpy", "sklearn", "open3d", "fast_pytorch_kmeans", "spconv", "torch_scatter", "huggingface_hub", "timm"]
out = {"python_modules": {m: bool(importlib.util.find_spec(m)) for m in mods}}
if out["python_modules"].get("torch"):
    import torch
    out["torch"] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
print(json.dumps(out, ensure_ascii=False, indent=2))
PY

printf 'sonata_repo=%s exists=%s\n' "${SONATA_REPO}" "$([ -d "${SONATA_REPO}/.git" ] && echo 1 || echo 0)"
printf 'pointcept_repo=%s exists=%s\n' "${POINTCEPT_REPO}" "$([ -d "${POINTCEPT_REPO}/.git" ] && echo 1 || echo 0)"
REMOTE
