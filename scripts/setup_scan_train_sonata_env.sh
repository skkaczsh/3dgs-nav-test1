#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-scan-train}"
SONATA_REPO="${SONATA_REPO:-/root/epfs/model_side_tracks/sonata}"
ENV_PREFIX="${ENV_PREFIX:-/root/epfs/conda_envs/sonata}"
TMUX_SESSION="${TMUX_SESSION:-scan_sonata_env_setup}"
RUN="${RUN:-0}"

echo "host=${SSH_HOST}"
echo "sonata_repo=${SONATA_REPO}"
echo "env_prefix=${ENV_PREFIX}"

if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to create/update Sonata conda env in tmux ${TMUX_SESSION}"
  exit 0
fi

ssh "${SSH_HOST}" bash -s -- "${SONATA_REPO}" "${ENV_PREFIX}" "${TMUX_SESSION}" <<'REMOTE'
set -euo pipefail
SONATA_REPO="$1"
ENV_PREFIX="$2"
TMUX_SESSION="$3"

test -f "${SONATA_REPO}/environment.yml"
mkdir -p "$(dirname "${ENV_PREFIX}")"
RUN_DIR="${ENV_PREFIX}_setup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/run.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a "${RUN_DIR}/setup.log") 2>&1
cd "${SONATA_REPO}"
conda env create -p "${ENV_PREFIX}" -f environment.yml --verbose
"${ENV_PREFIX}/bin/python" - <<'PY'
import importlib.util, json
mods = ["torch", "sonata", "spconv", "torch_scatter", "timm", "open3d", "fast_pytorch_kmeans"]
out = {"modules": {m: bool(importlib.util.find_spec(m)) for m in mods}}
if out["modules"].get("torch"):
    import torch
    out["torch"] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
date -Is > "${RUN_DIR}/DONE"
SCRIPT
chmod +x "${RUN_DIR}/run.sh"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" "${RUN_DIR}/run.sh"
echo "${RUN_DIR}"
tmux ls
REMOTE
