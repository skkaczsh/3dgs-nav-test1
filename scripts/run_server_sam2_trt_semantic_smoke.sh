#!/usr/bin/env bash
set -euo pipefail

# Run a small downstream semantic smoke using SAM2 TensorRT/C++ mask artifacts.
# This validates that compact RLE masks can pass through:
# sam2_qwen -> sky/adjacency merge -> prompt_v3 review -> completion.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MANIFEST="${MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_2000_2999.json}"
SAM_MASK_SOURCE="${SAM_MASK_SOURCE:-/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_rle50}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/sam2_tensorrt/semantic_eval_rle50_default_downstream10_cam0}"
FILTER_PREFIX="${FILTER_PREFIX:-cam0_}"
LIMIT="${LIMIT:-10}"
SHARDS="${SHARDS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
VLM_ENDPOINT="${VLM_ENDPOINT:-http://localhost:8001/v1/chat/completions}"
VLM_MODEL="${VLM_MODEL:-Qwen3.6-35B-A3B-Q4_K_M}"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/_smoke_work"

FILTERED_MANIFEST="${OUTPUT_DIR}/_smoke_work/manifest_filtered.json"
python3 - "${MANIFEST}" "${SAM_MASK_SOURCE}" "${FILTERED_MANIFEST}" "${FILTER_PREFIX}" "${LIMIT}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
sam_mask_source = Path(sys.argv[2])
output = Path(sys.argv[3])
prefix = sys.argv[4]
limit = int(sys.argv[5])

items = json.loads(manifest.read_text(encoding="utf-8")).get("items", [])
selected = []
for item in items:
    image_id = str(item.get("image_id", ""))
    if prefix and not image_id.startswith(prefix):
        continue
    if not (sam_mask_source / f"{image_id}_sam_masks.json").exists():
        continue
    selected.append(item)
    if len(selected) >= limit:
        break

output.write_text(json.dumps({"items": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "manifest": str(manifest),
    "sam_mask_source": str(sam_mask_source),
    "filtered_manifest": str(output),
    "filter_prefix": prefix,
    "limit": limit,
    "items": len(selected),
    "first": selected[0]["image_id"] if selected else None,
    "last": selected[-1]["image_id"] if selected else None,
}, ensure_ascii=False, indent=2))
if not selected:
    raise SystemExit("no matching SAM2 TensorRT masks found")
PY

MANIFEST="${FILTERED_MANIFEST}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
SAM_MASKS_DIR="${OUTPUT_DIR}/linked_sam_masks" \
EXISTING_SAM_DIR="${SAM_MASK_SOURCE}" \
PART0=/root/epfs/nonexistent \
PART1=/root/epfs/nonexistent \
START_INDEX=0 \
END_INDEX="${LIMIT}" \
SHARDS="${SHARDS}" \
CHUNK_SIZE="${CHUNK_SIZE}" \
MAX_TOKENS="${MAX_TOKENS}" \
VLM_ENDPOINT="${VLM_ENDPOINT}" \
VLM_MODEL="${VLM_MODEL}" \
bash "${SCRIPT_DIR}/run_server_semantic_completion_sharded.sh"

python3 - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

base = Path(sys.argv[1])
combo = "sam2_prompt_v3_sky_label_merge_completion"
counts = Counter()
frames = 0
semantic_png = 0
records = 0
for combo_dir in sorted((base / "images").glob(f"*/{combo}")):
    frames += 1
    semantic_png += int((combo_dir / "semantic.png").exists())
    label_records = combo_dir / "label_records.json"
    if not label_records.exists():
        continue
    data = json.loads(label_records.read_text(encoding="utf-8"))
    values = data.values() if isinstance(data, dict) else data
    for row in values:
        if isinstance(row, dict):
            counts[str(row.get("label", "unknown"))] += 1
            records += 1

summary = {
    "output_dir": str(base),
    "combo": combo,
    "frames": frames,
    "semantic_png": semantic_png,
    "label_records": records,
    "top_labels": counts.most_common(20),
}
(base / "sam2_trt_semantic_smoke_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
