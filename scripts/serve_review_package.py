#!/usr/bin/env python3
"""Serve the local dataset review package and PLY viewer."""

from __future__ import annotations

import argparse
import http.server
import json
import os
from functools import partial
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Users/skkac/Work/SCAN"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"missing root: {root}")
    os.chdir(root)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    urls = {
        "root": str(root),
        "qa_index": f"http://{args.host}:{args.port}/dataset_delivery_0000_0999/qa_index.html",
        "viewer": f"http://{args.host}:{args.port}/new_route/tools/semantic_ply_viewer.html",
        "package_manifest": f"http://{args.host}:{args.port}/dataset_delivery_0000_0999/package_manifest.json",
    }
    print(json.dumps(urls, indent=2), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
