#!/usr/bin/env python3
"""Compare frame-target geometry conflict diagnosis reports.

The target-diagnosis report is the correct gate for source-mask experiments:
if a mask tweak reduces 2D label pollution but increases target-level geometry
conflicts, it should not be promoted into the production route.
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
        return path.parent.name or path.stem, path
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


def as_counter(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(k): as_int(v) for k, v in value.items()}


def top_window_score(report: dict[str, Any]) -> int:
    windows = report.get("top_windows") or []
    if not isinstance(windows, list):
        return 0
    return sum(as_int(row.get("score_sum")) for row in windows if isinstance(row, dict))


def summarize(path: Path) -> dict[str, Any]:
    report = read_json(path)
    return {
        "path": str(path),
        "target_count": as_int(report.get("target_count")),
        "finding_count": as_int(report.get("finding_count")),
        "finding_points": sum(as_int(row.get("finding_points")) for row in report.get("top_windows", []) if isinstance(row, dict)),
        "top_window_score_sum": top_window_score(report),
        "finding_label_counts": as_counter(report.get("finding_label_counts")),
        "top_windows": report.get("top_windows") or [],
    }


def collect_keys(rows: dict[str, dict[str, Any]], field: str) -> list[str]:
    keys: set[str] = set()
    for row in rows.values():
        keys.update(row.get(field, {}))
    preferred = ["ground", "wall", "ceiling", "car", "railing", "grass", "unknown", "other"]
    out = [key for key in preferred if key in keys]
    out.extend(sorted(keys - set(out)))
    return out


def delta_counter(row: dict[str, int], base: dict[str, int], keys: list[str]) -> dict[str, int]:
    return {key: as_int(row.get(key, 0)) - as_int(base.get(key, 0)) for key in keys}


def build_comparison(named_reports: list[tuple[str, Path]]) -> dict[str, Any]:
    versions = {name: summarize(path) for name, path in named_reports}
    baseline_name = named_reports[0][0] if named_reports else ""
    baseline = versions.get(baseline_name, {})
    label_keys = collect_keys(versions, "finding_label_counts")
    deltas: dict[str, dict[str, Any]] = {}
    for name, row in versions.items():
        deltas[name] = {
            "target_count": as_int(row.get("target_count")) - as_int(baseline.get("target_count")),
            "finding_count": as_int(row.get("finding_count")) - as_int(baseline.get("finding_count")),
            "finding_points": as_int(row.get("finding_points")) - as_int(baseline.get("finding_points")),
            "top_window_score_sum": as_int(row.get("top_window_score_sum")) - as_int(baseline.get("top_window_score_sum")),
            "finding_label_counts": delta_counter(
                row.get("finding_label_counts", {}),
                baseline.get("finding_label_counts", {}),
                label_keys,
            ),
        }
    return {
        "baseline": baseline_name,
        "label_keys": label_keys,
        "versions": versions,
        "deltas_from_baseline": deltas,
    }


def markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Frame-Target Geometry Conflict Comparison",
        "",
        f"Baseline: `{comparison['baseline']}`",
        "",
        "| version | targets | findings | finding points | top-window score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in comparison["versions"].items():
        lines.append(
            f"| `{name}` | {row['target_count']:,} | {row['finding_count']:,} | "
            f"{row['finding_points']:,} | {row['top_window_score_sum']:,} |"
        )

    label_keys = comparison["label_keys"]
    lines.extend(["", "## Finding Label Counts", ""])
    lines.append("| version | " + " | ".join(label_keys) + " |")
    lines.append("| --- | " + " | ".join(["---:" for _ in label_keys]) + " |")
    for name, row in comparison["versions"].items():
        values = [f"`{name}`", *[f"{as_int(row['finding_label_counts'].get(key, 0)):,}" for key in label_keys]]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Delta From Baseline", ""])
    for name, row in comparison["deltas_from_baseline"].items():
        label_delta = ", ".join(
            f"{key}={as_int(row['finding_label_counts'].get(key, 0)):+,}"
            for key in label_keys
            if as_int(row["finding_label_counts"].get(key, 0))
        )
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- targets: {row['target_count']:+,}",
                f"- findings: {row['finding_count']:+,}",
                f"- finding points: {row['finding_points']:+,}",
                f"- top-window score: {row['top_window_score_sum']:+,}",
                f"- finding label delta: {label_delta or '0'}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Diagnosis report, optionally named as name=/path/geometry_conflicts_report.json. First report is baseline.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    comparison = build_comparison([parse_named_report(raw) for raw in args.report])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(comparison), encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": comparison["baseline"],
                "versions": list(comparison["versions"]),
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
