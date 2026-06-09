#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/epfs/third_party/ConceptSeg-R1}"
PYTHON="${PYTHON:-/root/epfs/conda_envs/conceptseg-r1/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/epfs/models/ConceptSeg-R1-7B}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-/root/epfs/new_route_stage1_skymask/conceptseg_problem_samples.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/new_route_stage1_skymask/conceptseg_smoke}"
LIMIT="${LIMIT:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_PIXELS="${MAX_PIXELS:-360000}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" - <<'PY' "${SAMPLE_MANIFEST}" "${OUTPUT_DIR}" "${LIMIT}"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
limit = int(sys.argv[3])

data = json.loads(manifest.read_text(encoding="utf-8"))
items = data.get("items", data if isinstance(data, list) else [])
if not items:
    raise SystemExit(f"no items in {manifest}")

selected = items[:limit]
runlist = output_dir / "runlist.json"
runlist.write_text(json.dumps({"items": selected}, indent=2), encoding="utf-8")
print(runlist)
PY

RUNLIST="${OUTPUT_DIR}/runlist.json"
"${PYTHON}" - <<'PY' "${RUNLIST}" "${OUTPUT_DIR}" "${REPO}" "${MODEL_PATH}" "${MAX_PIXELS}"
import json
import shlex
import subprocess
import sys
from pathlib import Path

runlist = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
repo = Path(sys.argv[3])
model_path = sys.argv[4]
max_pixels = sys.argv[5]

items = json.loads(runlist.read_text(encoding="utf-8"))["items"]
script = repo / "src/eval/inference_single_example.py"
rows = []
for item in items:
    image_id = item["image_id"]
    concept = item["concept"]
    image_path = item["image_path"]
    out_path = output_dir / f"{image_id}_{concept}.png"
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
            "image_id": image_id,
            "concept": concept,
            "image_path": image_path,
            "output_path": str(out_path),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    )
    if proc.returncode != 0:
        break

report = output_dir / "report.json"
report.write_text(json.dumps({"items": rows}, indent=2), encoding="utf-8")
print(f"wrote={report}")
if rows and rows[-1]["returncode"] != 0:
    raise SystemExit(rows[-1]["returncode"])
PY
