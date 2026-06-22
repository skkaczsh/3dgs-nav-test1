#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
POTREE_DIR="$ROOT_DIR/third_party/potree"
ZIP_PATH="${TMPDIR:-/tmp}/potree-develop.zip"

if [[ -d "$POTREE_DIR/build/potree" && -f "$POTREE_DIR/build/potree/potree.js" ]]; then
  echo "Potree assets already exist: $POTREE_DIR"
  exit 0
fi

mkdir -p "$ROOT_DIR/third_party"
rm -rf "$POTREE_DIR" "${TMPDIR:-/tmp}/potree-develop"

curl -L --fail --retry 3 \
  -o "$ZIP_PATH" \
  https://github.com/potree/potree/archive/refs/heads/develop.zip

unzip -q "$ZIP_PATH" -d "${TMPDIR:-/tmp}"
mv "${TMPDIR:-/tmp}/potree-develop" "$POTREE_DIR"

(
  cd "$POTREE_DIR"
  npm install
)

echo "Potree assets ready: $POTREE_DIR"
