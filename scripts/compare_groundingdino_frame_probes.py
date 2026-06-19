#!/usr/bin/env python3
"""Compare GroundingDINO frame-probe reports.

The detector probe is only useful if it gives narrow, stable candidate boxes.
This comparator makes that gate explicit before any detector output is allowed
to feed SAM or point-cloud projection.
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


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_counter(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): as_int(val) for key, val in value.items()}


def as_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): as_float(val) for key, val in value.items()}


def preferred_labels(rows: dict[str, dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows.values():
        keys.update(row.get("detection_counts", {}))
        keys.update(row.get("large_box_counts", {}))
        keys.update(row.get("mean_box_area_ratio_by_label", {}))
        keys.update(row.get("max_box_area_ratio_by_label", {}))
    preferred = ["railing", "car", "wall", "ground", "grass", "unknown", "other"]
    out = [key for key in preferred if key in keys]
    out.extend(sorted(keys - set(out)))
    return out


def summarize(path: Path) -> dict[str, Any]:
    report = read_json(path)
    detections = as_counter(report.get("detection_counts"))
    large = as_counter(report.get("large_box_counts"))
    large_rates = {
        label: large.get(label, 0) / max(detections.get(label, 0), 1)
        for label in set(detections) | set(large)
    }
    return {
        "path": str(path),
        "model": str(report.get("model", "")),
        "image_count": as_int(report.get("image_count")),
        "box_threshold": as_float(report.get("box_threshold")),
        "text_threshold": as_float(report.get("text_threshold")),
        "large_box_ratio": as_float(report.get("large_box_ratio")),
        "detection_counts": detections,
        "large_box_counts": large,
        "large_box_rates": large_rates,
        "mean_box_area_ratio_by_label": as_float_map(report.get("mean_box_area_ratio_by_label")),
        "max_box_area_ratio_by_label": as_float_map(report.get("max_box_area_ratio_by_label")),
    }


def int_delta(row: dict[str, int], base: dict[str, int], labels: list[str]) -> dict[str, int]:
    return {label: row.get(label, 0) - base.get(label, 0) for label in labels}


def float_delta(row: dict[str, float], base: dict[str, float], labels: list[str]) -> dict[str, float]:
    return {label: row.get(label, 0.0) - base.get(label, 0.0) for label in labels}


def build_comparison(named_reports: list[tuple[str, Path]]) -> dict[str, Any]:
    versions = {name: summarize(path) for name, path in named_reports}
    baseline_name = named_reports[0][0] if named_reports else ""
    baseline = versions.get(baseline_name, {})
    labels = preferred_labels(versions)
    deltas = {}
    for name, row in versions.items():
        deltas[name] = {
            "image_count": row.get("image_count", 0) - baseline.get("image_count", 0),
            "detection_counts": int_delta(row.get("detection_counts", {}), baseline.get("detection_counts", {}), labels),
            "large_box_counts": int_delta(row.get("large_box_counts", {}), baseline.get("large_box_counts", {}), labels),
            "large_box_rates": float_delta(row.get("large_box_rates", {}), baseline.get("large_box_rates", {}), labels),
            "mean_box_area_ratio_by_label": float_delta(
                row.get("mean_box_area_ratio_by_label", {}),
                baseline.get("mean_box_area_ratio_by_label", {}),
                labels,
            ),
        }
    return {
        "baseline": baseline_name,
        "labels": labels,
        "versions": versions,
        "deltas_from_baseline": deltas,
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def markdown(comparison: dict[str, Any]) -> str:
    labels = comparison["labels"]
    lines = [
        "# GroundingDINO Frame Probe Comparison",
        "",
        f"Baseline: `{comparison['baseline']}`",
        "",
        "| version | model | images | box th | text th |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, row in comparison["versions"].items():
        lines.append(
            f"| `{name}` | `{row['model']}` | {row['image_count']:,} | "
            f"{row['box_threshold']:.2f} | {row['text_threshold']:.2f} |"
        )

    for title, field, formatter in [
        ("Detection Counts", "detection_counts", lambda v: f"{as_int(v):,}"),
        ("Large Box Counts", "large_box_counts", lambda v: f"{as_int(v):,}"),
        ("Large Box Rates", "large_box_rates", lambda v: pct(as_float(v))),
        ("Mean Box Area Ratio", "mean_box_area_ratio_by_label", lambda v: pct(as_float(v))),
        ("Max Box Area Ratio", "max_box_area_ratio_by_label", lambda v: pct(as_float(v))),
    ]:
        lines.extend(["", f"## {title}", ""])
        lines.append("| version | " + " | ".join(labels) + " |")
        lines.append("| --- | " + " | ".join(["---:" for _ in labels]) + " |")
        for name, row in comparison["versions"].items():
            values = [f"`{name}`", *[formatter(row.get(field, {}).get(label, 0)) for label in labels]]
            lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Delta From Baseline", ""])
    for name, delta in comparison["deltas_from_baseline"].items():
        detection_delta = ", ".join(
            f"{label}={delta['detection_counts'][label]:+,}"
            for label in labels
            if delta["detection_counts"].get(label)
        )
        large_delta = ", ".join(
            f"{label}={delta['large_box_counts'][label]:+,}"
            for label in labels
            if delta["large_box_counts"].get(label)
        )
        rate_delta = ", ".join(
            f"{label}={pct(delta['large_box_rates'][label])}"
            for label in labels
            if abs(delta["large_box_rates"].get(label, 0.0)) > 1e-9
        )
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- images: {delta['image_count']:+,}",
                f"- detections: {detection_delta or '0'}",
                f"- large boxes: {large_delta or '0'}",
                f"- large-box rate delta: {rate_delta or '0'}",
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
        help="Probe report, optionally named as name=/path/groundingdino_frame_probe.json. First report is baseline.",
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
