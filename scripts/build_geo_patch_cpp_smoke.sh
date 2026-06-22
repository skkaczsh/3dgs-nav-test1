#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CXX="${CXX:-g++}"
SRC="${SRC:-${REPO_ROOT}/tools/geo_patch_region_model_smoke.cpp}"
OUT="${OUT:-${REPO_ROOT}/build/geo_patch/geo_patch_region_model_smoke}"

mkdir -p "$(dirname "${OUT}")"

"${CXX}" -std=c++17 -O2 -Wall -Wextra -pedantic "${SRC}" -o "${OUT}"
echo "${OUT}"
