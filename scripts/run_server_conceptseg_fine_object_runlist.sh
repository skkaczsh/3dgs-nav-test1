#!/usr/bin/env bash
set -euo pipefail

# Run ConceptSeg-R1 on a constrained fine-object runlist.
# Side-track only: this must not replace the main SAM2+Qwen semantic route.

BASE_DIR="${BASE_DIR:-/root/epfs/model_side_tracks}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/ConceptSeg-R1}"
MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/ConceptSeg-R1-7B}"
CONDA_ENV="${CONDA_ENV:-/root/epfs/conda_envs/conceptseg-r1}"
SAM3_DIR="${SAM3_DIR:-/root/epfs/third_party/ConceptSeg-R1/sam3-main}"
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-/root/epfs/new_route_stage1_skymask/conceptseg_fine_object_runlist_v008/runlist.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/new_route_stage1_skymask/conceptseg_fine_object_runlist_v008_outputs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MAX_PIXELS="${MAX_PIXELS:-240000}"
LIMIT="${LIMIT:-12}"
START_INDEX="${START_INDEX:-0}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "Missing ConceptSeg-R1 repo: ${REPO_DIR}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing ConceptSeg-R1 model: ${MODEL_PATH}" >&2
  exit 3
fi
if [[ ! -f "${SAMPLE_MANIFEST}" ]]; then
  echo "Missing runlist: ${SAMPLE_MANIFEST}" >&2
  exit 4
fi

PYTHON_BIN="${CONDA_ENV}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

if [[ ! -d "${SAM3_DIR}/sam3" ]]; then
  if [[ -d "${REPO_DIR}/sam3-main/sam3" ]]; then
    SAM3_DIR="${REPO_DIR}/sam3-main"
  else
    echo "Missing modified SAM3 source directory: ${SAM3_DIR}" >&2
    exit 5
  fi
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${SAM3_DIR}:${REPO_DIR}/src/open-r1-multimodal/src:${REPO_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/epfs/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

INFERENCE_SCRIPT="${REPO_DIR}/src/eval/inference_single_example.py"
if [[ "${ATTENTION_IMPLEMENTATION}" != "flash_attention_2" ]]; then
  TMP_SCRIPT="${OUTPUT_DIR}/inference_single_example_${ATTENTION_IMPLEMENTATION}.py"
  "${PYTHON_BIN}" - "${INFERENCE_SCRIPT}" "${TMP_SCRIPT}" "${ATTENTION_IMPLEMENTATION}" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
attn = sys.argv[3]
text = src.read_text(encoding="utf-8")
text = text.replace('attn_implementation="flash_attention_2"', f'attn_implementation="{attn}"')
dst.write_text(text, encoding="utf-8")
PY
  INFERENCE_SCRIPT="${TMP_SCRIPT}"
fi

"${PYTHON_BIN}" - <<'PY' "${SAMPLE_MANIFEST}" "${OUTPUT_DIR}" "${LIMIT}" "${START_INDEX}" "${REPO_DIR}" "${MODEL_PATH}" "${MAX_PIXELS}" "${INFERENCE_SCRIPT}"
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
limit = int(sys.argv[3])
start_index = int(sys.argv[4])
repo = Path(sys.argv[5])
model_path = sys.argv[6]
max_pixels = sys.argv[7]
script = Path(sys.argv[8])

data = json.loads(manifest.read_text(encoding="utf-8"))
items = data.get("items", data if isinstance(data, list) else [])
selected = items[start_index : start_index + limit if limit >= 0 else None]
if not selected:
    raise SystemExit(f"no selected items start={start_index} limit={limit} in {manifest}")

def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_")[:180] or "item"

rows = []
for offset, item in enumerate(selected):
    absolute_index = start_index + offset
    image_id = item["image_id"]
    concept = item["concept"]
    image_path = item["image_path"]
    out_path = output_dir / f"{absolute_index:04d}_{slug(image_id)}.png"
    cmd = [
        sys.executable,
        str(script),
        "--model_path",
        model_path,
        "--infer_path",
        image_path,
        "--question",
        concept,
        "--output_path",
        str(out_path),
        "--max_pixels",
        max_pixels,
    ]
    print("+", " ".join(shlex.quote(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True)
    rows.append(
        {
            "absolute_index": absolute_index,
            "image_id": image_id,
            "concept": concept,
            "image_path": image_path,
            "output_path": str(out_path),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
            "metadata": item.get("metadata", {}),
        }
    )
    report = output_dir / "report.json"
    report.write_text(json.dumps({"source_manifest": str(manifest), "items": rows}, indent=2), encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

print(f"wrote={output_dir / 'report.json'}")
PY

echo "output_dir: ${OUTPUT_DIR}"
