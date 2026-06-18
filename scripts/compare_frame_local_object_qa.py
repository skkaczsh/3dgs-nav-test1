#!/usr/bin/env python3
"""Compare multiple frame-local object QA reports.

The route-level signal is ``all_risk_reason_counts``.  The older
``risk_reason_counts`` field is retained by the QA pack for the selected review
candidate set only, so this script deliberately reports both full-object risk
and review-candidate risk to avoid mixing the two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_named_report(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.stem, path
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"missing report name in {raw!r}")
    return name, Path(path)


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def risk_counts(report: dict[str, Any], full: bool) -> dict[str, int]:
    if full:
        data = report.get("all_risk_reason_counts") or {}
    else:
        data = report.get("candidate_risk_reason_counts") or report.get("risk_reason_counts") or {}
    if not isinstance(data, dict):
        return {}
    return {str(k): as_int(v) for k, v in data.items()}


def build_comparison(named_reports: list[tuple[str, Path]]) -> dict[str, Any]:
    versions: dict[str, dict[str, Any]] = {}
    risk_keys: set[str] = set()
    candidate_risk_keys: set[str] = set()
    baseline_name = named_reports[0][0] if named_reports else ""
    baseline_full: dict[str, int] = {}

    for name, path in named_reports:
        report = read_json(path)
        full_counts = risk_counts(report, full=True)
        candidate_counts = risk_counts(report, full=False)
        risk_keys.update(full_counts)
        candidate_risk_keys.update(candidate_counts)
        if name == baseline_name:
            baseline_full = full_counts
        versions[name] = {
            "path": str(path),
            "objects": as_int(report.get("objects")),
            "semantic_label_counts": report.get("semantic_label_counts") or {},
            "all_candidate_count": as_int(report.get("all_candidate_count")),
            "candidate_count": as_int(report.get("candidate_count")),
            "all_risk_reason_counts": full_counts,
            "candidate_risk_reason_counts": candidate_counts,
        }

    deltas_from_baseline: dict[str, dict[str, int]] = {}
    for name, row in versions.items():
        full_counts = row["all_risk_reason_counts"]
        deltas_from_baseline[name] = {
            key: as_int(full_counts.get(key, 0)) - as_int(baseline_full.get(key, 0))
            for key in sorted(risk_keys)
        }

    return {
        "baseline": baseline_name,
        "versions": versions,
        "full_risk_keys": sorted(risk_keys),
        "candidate_risk_keys": sorted(candidate_risk_keys),
        "all_risk_deltas_from_baseline": deltas_from_baseline,
    }


def markdown_table(comparison: dict[str, Any]) -> str:
    versions = comparison["versions"]
    risk_keys = comparison["full_risk_keys"]
    columns = ["version", "objects", "all risky objects", *risk_keys]
    lines = [
        "# Frame-Local Object QA Comparison",
        "",
        f"Baseline: `{comparison['baseline']}`",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---", *["---:" for _ in columns[1:]]]) + " |",
    ]
    for name, row in versions.items():
        full_counts = row["all_risk_reason_counts"]
        values = [
            f"`{name}`",
            f"`{row['objects']}`",
            f"`{row['all_candidate_count']}`",
            *[f"`{as_int(full_counts.get(key, 0))}`" for key in risk_keys],
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Full-Risk Delta From Baseline", ""])
    lines.append("| " + " | ".join(["version", *risk_keys]) + " |")
    lines.append("| " + " | ".join(["---", *["---:" for _ in risk_keys]]) + " |")
    for name, deltas in comparison["all_risk_deltas_from_baseline"].items():
        values = [f"`{name}`", *[f"`{deltas.get(key, 0):+d}`" for key in risk_keys]]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append(
        "Note: full-risk fields use `all_risk_reason_counts`; selected review-candidate counts are intentionally not used as route-level metrics."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Report path, optionally named as name=/path/frame_local_object_qa_report.json. First report is baseline.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    named_reports = [parse_named_report(raw) for raw in args.report]
    comparison = build_comparison(named_reports)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_table(comparison), encoding="utf-8")
    print(json.dumps({"versions": list(comparison["versions"]), "output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
