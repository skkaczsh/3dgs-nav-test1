#!/usr/bin/env bash
# Build one validated depth-aware evidence view plan from deterministic shards.
set -euo pipefail

usage() {
  echo "usage: $0 --data-dir DIR --objects-jsonl FILE --object-ply FILE --frame-root DIR --depth-map-dir DIR --sky-mask-dir DIR --output-dir DIR [--workers N]" >&2
  exit 2
}

DATA_DIR=""
OBJECTS_JSONL=""
OBJECT_PLY=""
FRAME_ROOT=""
DEPTH_MAP_DIR=""
SKY_MASK_DIR=""
OUTPUT_DIR=""
WORKERS=4
PREFILTER=80
MAX_FRAMES=12
FRAME_STRIDE=10
DEPTH_TOLERANCE=0.2
DEPTH_NEIGHBORHOOD=1
MIN_PROJECTED=20
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --objects-jsonl) OBJECTS_JSONL="$2"; shift 2 ;;
    --object-ply) OBJECT_PLY="$2"; shift 2 ;;
    --frame-root) FRAME_ROOT="$2"; shift 2 ;;
    --depth-map-dir) DEPTH_MAP_DIR="$2"; shift 2 ;;
    --sky-mask-dir) SKY_MASK_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --prefilter) PREFILTER="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --frame-stride) FRAME_STRIDE="$2"; shift 2 ;;
    --depth-tolerance) DEPTH_TOLERANCE="$2"; shift 2 ;;
    --depth-neighborhood) DEPTH_NEIGHBORHOOD="$2"; shift 2 ;;
    --min-projected-points) MIN_PROJECTED="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$DATA_DIR" && -n "$OBJECTS_JSONL" && -n "$OBJECT_PLY" && -n "$FRAME_ROOT" && -n "$DEPTH_MAP_DIR" && -n "$SKY_MASK_DIR" && -n "$OUTPUT_DIR" ]] || usage
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "--workers must be a positive integer" >&2; exit 2; }

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SHARD_DIR="$OUTPUT_DIR/shards"
mkdir -p "$SHARD_DIR"

"$PYTHON_BIN" - "$OBJECTS_JSONL" "$SHARD_DIR" "$WORKERS" <<'PY'
import json
import sys
from pathlib import Path

source, target, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
if not rows:
    raise SystemExit("objects JSONL is empty")
shards = [[] for _ in range(workers)]
for index, row in enumerate(rows):
    shards[index % workers].append(row)
for index, shard in enumerate(shards):
    (target / f"candidates_{index:02d}.jsonl").write_text(
        chr(10).join(json.dumps(row, ensure_ascii=False) for row in shard) + chr(10), encoding="utf-8",
    )
print(json.dumps({"objects": len(rows), "workers": workers, "shard_sizes": [len(shard) for shard in shards]}))
PY

pids=()
for (( shard=0; shard<WORKERS; shard++ )); do
  shard_dir="$SHARD_DIR/$shard"
  mkdir -p "$shard_dir"
  (
    # Each NumPy projection worker receives a bounded CPU share.  More threads
    # per worker just creates BLAS contention when the runner is sharded.
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/build_object_image_evidence.py" \
      --data-dir "$DATA_DIR" \
      --objects-jsonl "$SHARD_DIR/candidates_$(printf '%02d' "$shard").jsonl" \
      --object-ply "$OBJECT_PLY" \
      --frame-root "$FRAME_ROOT" \
      --output-dir "$shard_dir" \
      --global-visibility \
      --global-depth-map-dir "$DEPTH_MAP_DIR" \
      --sky-mask-dir "$SKY_MASK_DIR" \
      --global-view-plan "$shard_dir/global_view_plan.json" \
      --global-view-plan-depth-aware \
      --global-view-plan-prefilter "$PREFILTER" \
      --max-frame-pool "$MAX_FRAMES" \
      --view-selection projected \
      --frame-stride "$FRAME_STRIDE" \
      --depth-tolerance "$DEPTH_TOLERANCE" \
      --depth-neighborhood "$DEPTH_NEIGHBORHOOD" \
      --min-projected-points "$MIN_PROJECTED" \
      --max-depth-cache-entries 256 \
      --progress-every 25 \
      >"$shard_dir/plan.stdout.json" 2>"$shard_dir/plan.stderr.log"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON_BIN" - "$SHARD_DIR" "$OUTPUT_DIR/global_view_plan_depth_aware.json" "$WORKERS" <<'PY'
import json
import sys
from pathlib import Path

shard_dir, output, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = []
metadata = None
for index in range(workers):
    data = json.loads((shard_dir / str(index) / "global_view_plan.json").read_text(encoding="utf-8"))
    if data.get("schema") != "global-evidence-view-plan/v1":
        raise SystemExit(f"unsupported shard plan schema in shard {index}")
    current = {key: data.get(key) for key in ("frame_stride", "view_selection", "max_frame_pool", "depth_aware", "prefilter_frames")}
    if metadata is None:
        metadata = current
    elif current != metadata:
        raise SystemExit(f"inconsistent planning parameters in shard {index}")
    rows.extend(data.get("objects") or [])
object_ids = [int(row["object_id"]) for row in rows]
if len(object_ids) != len(set(object_ids)):
    raise SystemExit("duplicate object ids across shards")
rows.sort(key=lambda row: int(row["object_id"]))
selected = sorted({int(frame_id) for row in rows for frame_id in row.get("frame_ids") or []})
output.write_text(json.dumps({
    "schema": "global-evidence-view-plan/v1",
    **(metadata or {}),
    "object_count": len(rows),
    "selected_frame_count": len(selected),
    "selected_frame_ids": selected,
    "objects": rows,
}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
print(json.dumps({"objects": len(rows), "selected_frames": len(selected), "output": str(output)}))
PY
