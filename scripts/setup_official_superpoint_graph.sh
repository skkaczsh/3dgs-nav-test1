#!/usr/bin/env bash
set -euo pipefail

# Build the original Superpoint Graph handcrafted partition backend for the
# active Python interpreter. This is an isolated third-party asset: project
# code only consumes its two compiled extension modules through an explicit
# --superpoint-graph-root / SUPERPOINT_GRAPH_ROOT contract.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPG_ROOT="${SUPERPOINT_GRAPH_ROOT:-${REPO_ROOT}/third_party/superpoint_graph}"
SPG_BRANCH="${SPG_BRANCH:-ssp+spg}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS="${JOBS:-8}"

if [[ ! -d "${SPG_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${SPG_ROOT}")"
  git clone --depth 1 --branch "${SPG_BRANCH}" --recurse-submodules \
    https://github.com/loicland/superpoint_graph.git "${SPG_ROOT}"
else
  git -C "${SPG_ROOT}" submodule update --init --recursive
fi

PARTITION="${SPG_ROOT}/partition"
PY_INCLUDE="$(${PYTHON_BIN} -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PY_NUMPY_INCLUDE="$(${PYTHON_BIN} -c 'import numpy; print(numpy.get_include())')"
PY_LIBRARY="$(${PYTHON_BIN} - <<'PY'
import glob
import os
import sys
import sysconfig

libdir = sysconfig.get_config_var("LIBDIR")
version = f"{sys.version_info.major}.{sys.version_info.minor}"
candidates = sorted(glob.glob(os.path.join(libdir, f"libpython{version}.so*")))
if not candidates:
    raise SystemExit(f"no shared libpython{version}.so found in {libdir}")
print(candidates[-1])
PY
)"

cmake -S "${PARTITION}/ply_c" -B "${PARTITION}/ply_c/build-py" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
  -DPYTHON_INCLUDE_DIR="${PY_INCLUDE}" \
  -DPYTHON_LIBRARY="${PY_LIBRARY}" \
  -DPYTHON_NUMPY_INCLUDE_DIR="${PY_NUMPY_INCLUDE}" \
  -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3
cmake --build "${PARTITION}/ply_c/build-py" -j"${JOBS}"

cmake -S "${PARTITION}/cut-pursuit" -B "${PARTITION}/cut-pursuit/build-py" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
  -DPYTHON_INCLUDE_DIR="${PY_INCLUDE}" \
  -DPYTHON_LIBRARY="${PY_LIBRARY}" \
  -DPYTHON_NUMPY_INCLUDE_DIR="${PY_NUMPY_INCLUDE}"
cmake --build "${PARTITION}/cut-pursuit/build-py" -j"${JOBS}"

ln -sfn "ply_c/build-py/libply_c.so" "${PARTITION}/libply_c.so"
ln -sfn "cut-pursuit/build-py/src/libcp.so" "${PARTITION}/libcp.so"

PYTHONPATH="${PARTITION}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<'PY'
import numpy as np
import graphs
import libcp
import libply_c

xyz = np.ascontiguousarray(np.array([
    [0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.06, 0.0, 0.0], [0.0, 0.03, 0.0],
], dtype=np.float32))
graph, target = graphs.compute_graph_nn_2(xyz, 2, 3)
geof = libply_c.compute_geof(xyz, target, 3).astype("float32")
_components, labels = libcp.cutpursuit(
    geof, graph["source"], graph["target"], np.ones(len(graph["source"]), dtype="float32"), 0.1,
)
assert len(labels) == len(xyz)
print({"official_spg_smoke": "passed", "geof_shape": tuple(geof.shape)})
PY

git -C "${SPG_ROOT}" rev-parse --short HEAD
