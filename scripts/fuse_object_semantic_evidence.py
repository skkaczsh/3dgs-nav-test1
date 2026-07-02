#!/usr/bin/env python3
"""Fuse object-level semantic evidence without changing object ownership."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.semantic_evidence_fusion import apply_decision, choose_label, params_from_args, summarize


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    params = params_from_args(args)
    objects = read_jsonl(args.objects_jsonl)
    output = [apply_decision(row, choose_label(row, params)) for row in objects]
    report = {
        "schema": "object-semantic-evidence-fusion/v1",
        "objects_jsonl": str(args.objects_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "params": serializable_args(args),
        **summarize(output),
    }
    write_jsonl(args.output_jsonl, output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sam-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=1.25)
    parser.add_argument("--scene-weight", type=float, default=0.35)
    parser.add_argument("--min-total-weight", type=float, default=3.0)
    parser.add_argument("--min-winner-ratio", type=float, default=0.58)
    parser.add_argument("--min-scene-supported-ratio", type=float, default=0.52)
    parser.add_argument("--allow-scene-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
