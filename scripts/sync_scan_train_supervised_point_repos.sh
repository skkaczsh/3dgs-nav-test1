#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-scan-train}"
BASE_DIR="${BASE_DIR:-/root/epfs/model_side_tracks}"
RUN="${RUN:-0}"

echo "host=${SSH_HOST}"
echo "base_dir=${BASE_DIR}"

if [[ "${RUN}" != "1" ]]; then
  echo "dry_run=1"
  echo "set RUN=1 to clone/update Sonata and Pointcept repos under ${BASE_DIR}"
  exit 0
fi

ssh "${SSH_HOST}" bash -s -- "${BASE_DIR}" <<'REMOTE'
set -euo pipefail
BASE_DIR="$1"
mkdir -p "${BASE_DIR}"
cd "${BASE_DIR}"

if [[ -d sonata/.git ]]; then
  git -C sonata pull --ff-only
else
  git clone https://github.com/facebookresearch/sonata.git sonata
fi

if [[ -d pointcept/.git ]]; then
  git -C pointcept pull --ff-only
else
  git clone https://github.com/Pointcept/Pointcept.git pointcept
fi

git -C sonata rev-parse --short HEAD
git -C pointcept rev-parse --short HEAD
REMOTE
