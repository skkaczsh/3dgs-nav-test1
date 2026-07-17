#!/usr/bin/env bash
# Retry only Superpoints lacking geometry-gated image evidence using projected visibility.
set -euo pipefail

usage() {
  echo "usage: $0 --base DIR --data-dir DIR --frame-root DIR --depth-map-dir DIR --sky-mask-dir DIR" >&2
  exit 2
}

BASE=""
DATA_DIR=""
FRAME_ROOT=""
DEPTH_MAP_DIR=""
SKY_MASK_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --frame-root) FRAME_ROOT="$2"; shift 2 ;;
    --depth-map-dir) DEPTH_MAP_DIR="$2"; shift 2 ;;
    --sky-mask-dir) SKY_MASK_DIR="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$BASE" && -n "$DATA_DIR" && -n "$FRAME_ROOT" && -n "$DEPTH_MAP_DIR" && -n "$SKY_MASK_DIR" ]] || usage

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASELINE="$BASE/evidence_full"
RETRY="$BASE/evidence_retry_projected"
MANIFEST="$BASE/retry_projected_candidates.jsonl"

"$PYTHON_BIN" - "$BASE" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
objects = [json.loads(line) for line in (base / "samples" / "objects.jsonl").read_text(encoding="utf-8").splitlines() if line]
evidence = base / "evidence_full" / "object_image_evidence.jsonl"
seen = {json.loads(line)["object_id"] for line in evidence.read_text(encoding="utf-8").splitlines() if line}
missing = [row for row in objects if row["object_id"] not in seen]
target = Path(sys.argv[2])
with target.open("w", encoding="utf-8") as handle:
    for row in missing:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"retry_candidates": len(missing), "output": str(target)}))
PY

rm -rf "$RETRY"
PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/build_object_image_evidence.py" \
  --data-dir "$DATA_DIR" \
  --objects-jsonl "$MANIFEST" \
  --object-ply "$BASE/samples/object_samples.ply" \
  --frame-root "$FRAME_ROOT" \
  --output-dir "$RETRY" \
  --global-visibility \
  --global-depth-map-dir "$DEPTH_MAP_DIR" \
  --sky-mask-dir "$SKY_MASK_DIR" \
  --start 0 --end 6180 --frame-stride 10 \
  --view-selection projected --max-frame-pool 12 --top-k 2 \
  --max-points-per-object 1500 --min-projected-points 20 \
  --depth-tolerance 0.20 --depth-neighborhood 1 \
  --bbox-percentile 2 --min-bbox-area 900 --score-mode tight \
  --save-projected-samples 128
