#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build_potree_from_ply.sh INPUT_ASCII_PLY OUTPUT_DIR [POTREE_CONVERTER]

Converts project ASCII PLY -> LAS -> Potree 2.x octree.

Environment:
  PYTHON                 Python with numpy + laspy (default: /opt/anaconda3/bin/python if present)
  POTREE_CONVERTER       PotreeConverter binary path, overrides arg 3

Example:
  scripts/build_potree_from_ply.sh \
    server_parking_priority_s10/run/geo_patches_random_color.ply \
    server_parking_priority_s10/run/potree
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
  usage
  exit 0
fi

INPUT_PLY="$1"
OUTPUT_DIR="$2"
POTREE_CONVERTER="${POTREE_CONVERTER:-${3:-}}"
if [[ -z "$POTREE_CONVERTER" ]]; then
  if command -v PotreeConverter >/dev/null 2>&1; then
    POTREE_CONVERTER="$(command -v PotreeConverter)"
  elif [[ -x "/root/epfs/SCAN/third_party/PotreeConverter-2.1.2/build/PotreeConverter" ]]; then
    POTREE_CONVERTER="/root/epfs/SCAN/third_party/PotreeConverter-2.1.2/build/PotreeConverter"
  else
    echo "ERROR: PotreeConverter not found. Pass it as arg 3 or POTREE_CONVERTER=..." >&2
    exit 2
  fi
fi

if [[ -x "/opt/anaconda3/bin/python" ]]; then
  PYTHON="${PYTHON:-/opt/anaconda3/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

mkdir -p "$OUTPUT_DIR"
TMP_DIR="$OUTPUT_DIR/_tmp"
mkdir -p "$TMP_DIR"
LAS_PATH="$TMP_DIR/input.las"

"$PYTHON" "$(dirname "$0")/convert_ascii_ply_to_las.py" \
  --input-ply "$INPUT_PLY" \
  --output-las "$LAS_PATH"

rm -rf "$OUTPUT_DIR/data"
"$POTREE_CONVERTER" "$LAS_PATH" -o "$OUTPUT_DIR/data" -m random

cat > "$OUTPUT_DIR/potree_manifest.json" <<EOF
{
  "input_ply": "$INPUT_PLY",
  "las": "$LAS_PATH",
  "metadata": "$OUTPUT_DIR/data/metadata.json",
  "converter": "$POTREE_CONVERTER"
}
EOF

echo "Potree output: $OUTPUT_DIR/data/metadata.json"
