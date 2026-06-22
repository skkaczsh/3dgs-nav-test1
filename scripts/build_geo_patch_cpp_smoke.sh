#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CXX="${CXX:-g++}"
SMOKE_SRC="${SMOKE_SRC:-${REPO_ROOT}/tools/geo_patch_region_model_smoke.cpp}"
SMOKE_OUT="${SMOKE_OUT:-${REPO_ROOT}/build/geo_patch/geo_patch_region_model_smoke}"
GROWER_SRC="${GROWER_SRC:-${REPO_ROOT}/tools/geo_patch_region_grower.cpp}"
GROWER_OUT="${GROWER_OUT:-${REPO_ROOT}/build/geo_patch/geo_patch_region_grower}"

mkdir -p "$(dirname "${SMOKE_OUT}")"

"${CXX}" -std=c++17 -O2 -Wall -Wextra -pedantic "${SMOKE_SRC}" -o "${SMOKE_OUT}"
"${CXX}" -std=c++17 -O2 -Wall -Wextra -pedantic "${GROWER_SRC}" -o "${GROWER_OUT}"
echo "${SMOKE_OUT}"
echo "${GROWER_OUT}"
