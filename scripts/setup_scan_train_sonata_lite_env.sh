#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-scan-train}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/envs/depth-anything-3/bin/python}"
VENV_DIR="${VENV_DIR:-/root/epfs/venvs/sonata-lite}"
SONATA_REPO="${SONATA_REPO:-/root/epfs/model_side_tracks/sonata}"
RUN="${RUN:-0}"

echo "host=${SSH_HOST}"
echo "base_python=${BASE_PYTHON}"
echo "venv_dir=${VENV_DIR}"

if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to create/update the lightweight Sonata venv"
  exit 0
fi

ssh "${SSH_HOST}" bash -s -- "${BASE_PYTHON}" "${VENV_DIR}" "${SONATA_REPO}" <<'REMOTE'
set -euo pipefail
BASE_PYTHON="$1"
VENV_DIR="$2"
SONATA_REPO="$3"

test -x "${BASE_PYTHON}"
test -d "${SONATA_REPO}"
"${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install -U pip
"${VENV_DIR}/bin/python" -m pip install \
  addict timm fast_pytorch_kmeans spconv-cu118 \
  torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu118.html

PYTHONPATH="${SONATA_REPO}:${PYTHONPATH:-}" "${VENV_DIR}/bin/python" - <<'PY'
import importlib.util, json
mods = ["torch", "sonata", "spconv", "torch_scatter", "timm", "open3d", "fast_pytorch_kmeans"]
out = {"modules": {m: bool(importlib.util.find_spec(m)) for m in mods}}
if out["modules"]["torch"]:
    import torch
    out["torch"] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
print(json.dumps(out, ensure_ascii=False, indent=2))
missing = [m for m, ok in out["modules"].items() if not ok]
raise SystemExit(1 if missing else 0)
PY
REMOTE
