#!/usr/bin/env python3
"""Split a semantic manifest into N round-robin shard manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--prefix", default="shard")
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = data.get("items", [])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for shard in range(args.shards):
        shard_items = items[shard::args.shards]
        path = args.output_dir / f"{args.prefix}_{shard:02d}.json"
        path.write_text(json.dumps({"items": shard_items}, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"shard": shard, "path": str(path), "items": len(shard_items)})
    print(json.dumps({"input": str(args.manifest), "items": len(items), "shards": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
