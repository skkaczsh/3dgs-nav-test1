#!/usr/bin/env python3
"""Build a lightweight review page for selected semantic viewer objects."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABEL_ZH = {
    "ground": "地面",
    "floor": "地面",
    "wall": "墙面",
    "grass": "草地",
    "car": "汽车",
    "railing": "栏杆/护栏",
    "unknown": "未知",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def viewer_id(row: dict[str, Any]) -> int | str:
    value = row.get("viewer_object_id", row.get("object_id"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def point_count(row: dict[str, Any]) -> int:
    try:
        return int(row.get("point_count") or 0)
    except (TypeError, ValueError):
        return 0


def label(row: dict[str, Any]) -> str:
    return str(row.get("semantic_label") or "unknown")


def zh(value: str) -> str:
    return LABEL_ZH.get(value, value)


def pick_review_objects(rows: list[dict[str, Any]], per_label: int) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[label(row)].append(row)

    selected: dict[str, dict[str, Any]] = {}
    for key in ["car", "railing", "wall", "ground", "grass", "unknown"]:
        for row in sorted(by_label.get(key, []), key=point_count, reverse=True)[:per_label]:
            selected[str(viewer_id(row))] = row

    for row in rows:
        status = str(row.get("status") or "")
        if "local_geometry" in status or status.startswith("priority_"):
            selected[str(viewer_id(row))] = row

    return sorted(selected.values(), key=lambda row: (label(row), -point_count(row), str(viewer_id(row))))


def make_viewer_url(args: argparse.Namespace, object_id: int | str, mode: str) -> str:
    url = (
        f"{args.viewer_path}?file={args.ply_url}&objects={args.objects_url}"
        f"&mode={mode}&stride=1&pointSize=2&object={object_id}"
    )
    return url


def make_full_viewer_url(args: argparse.Namespace, mode: str) -> str:
    url = f"{args.viewer_path}?file={args.ply_url}&objects={args.objects_url}&mode={mode}&stride=1&pointSize=1.5"
    return url


def object_summary(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    oid = viewer_id(row)
    return {
        "object_id": oid,
        "source_object_id": row.get("object_id"),
        "label": label(row),
        "label_zh": zh(label(row)),
        "status": row.get("status"),
        "point_count": point_count(row),
        "target_count": row.get("target_count"),
        "frames": row.get("frames"),
        "bbox_3d": row.get("bbox_3d"),
        "centroid": row.get("centroid"),
        "normal": row.get("normal"),
        "label_votes": row.get("label_votes"),
        "semantic_url": make_viewer_url(args, oid, "semantic"),
        "object_url": make_viewer_url(args, oid, "object"),
        "rgb_url": make_viewer_url(args, oid, "rgb"),
    }


def write_decision_template(objects: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "object_id",
        "source_object_id",
        "current_label",
        "decision",
        "new_label",
        "confidence",
        "reviewer",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in objects:
            writer.writerow(
                {
                    "object_id": item["object_id"],
                    "source_object_id": item.get("source_object_id") or "",
                    "current_label": item.get("label") or "unknown",
                    "decision": "pending",
                    "new_label": "",
                    "confidence": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def render_html(report: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value if value is not None else ""))

    rows = []
    for item in report["objects"]:
        rows.append(
            "<tr>"
            f"<td><code>{esc(item['object_id'])}</code><br><span class='muted'>{esc(item.get('source_object_id'))}</span></td>"
            f"<td>{esc(item['label_zh'])}<br><span class='muted'>{esc(item['label'])}</span></td>"
            f"<td>{esc(item.get('status'))}</td>"
            f"<td>{int(item.get('point_count') or 0):,}</td>"
            f"<td>{esc(item.get('target_count'))}</td>"
            f"<td><code>{esc(item.get('frames'))}</code></td>"
            f"<td><a href='{esc(item['semantic_url'])}' target='_blank'>semantic</a> "
            f"<a href='{esc(item['object_url'])}' target='_blank'>object</a> "
            f"<a href='{esc(item['rgb_url'])}' target='_blank'>rgb</a></td>"
            "</tr>"
        )
    css = """
    :root { color-scheme: dark; --bg:#111318; --panel:#181b22; --line:#303642; --text:#eef1f5; --muted:#9aa3af; --accent:#42b8d8; }
    body { margin:0; padding:18px; background:var(--bg); color:var(--text); font:13px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    h1 { margin:0 0 8px; font-size:18px; }
    .muted { color:var(--muted); }
    code { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
    table { width:100%; border-collapse:collapse; margin-top:14px; }
    th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:var(--muted); background:#151922; position:sticky; top:0; }
    a { color:var(--accent); margin-right:8px; }
    .panel { border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:12px; margin:12px 0; }
    """
    decision_url = report.get("decision_template_url") or report.get("decision_template") or ""
    decision_line = (
        f'<div>Manual decisions: <a href="{esc(decision_url)}" target="_blank">{esc(decision_url)}</a></div>'
        if decision_url
        else ""
    )
    full_urls = report.get("full_viewer_urls") or {}
    full_links = " ".join(
        f'<a href="{esc(url)}" target="_blank">{esc(label)}</a>'
        for label, url in [
            ("整版语义", full_urls.get("semantic")),
            ("整版 Object", full_urls.get("object")),
            ("整版 RGB", full_urls.get("rgb")),
        ]
        if url
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>{esc(report['title'])}</title><style>{css}</style></head>
<body>
  <h1>{esc(report['title'])}</h1>
  <div class="panel">
    <div>Generated: <code>{esc(report['generated_at'])}</code></div>
    <div>Objects: <code>{len(report['objects'])}</code></div>
    <div>Viewer index: <a href="{esc(report['viewer_index_url'])}" target="_blank">{esc(report['viewer_index_url'])}</a></div>
    <div>Full version: {full_links}</div>
    {decision_line}
  </div>
  <table>
    <thead><tr><th>Object</th><th>Label</th><th>Status</th><th>Points</th><th>Targets</th><th>Frames</th><th>Open</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Pure Surface Visibility Object Review")
    parser.add_argument("--per-label", type=int, default=8)
    parser.add_argument("--viewer-path", default="/tools/semantic_ply_viewer.html")
    parser.add_argument("--viewer-index-url", default="/tools/semantic_viewer_index.html")
    parser.add_argument("--ply-url", required=True)
    parser.add_argument("--objects-url", required=True)
    parser.add_argument("--decision-template-url", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.objects_jsonl)
    objects = [object_summary(row, args) for row in pick_review_objects(rows, args.per_label)]
    report = {
        "schema": "semantic-object-review-index/v1",
        "title": args.title,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "objects_jsonl": str(args.objects_jsonl),
        "viewer_index_url": args.viewer_index_url,
        "full_viewer_urls": {
            "semantic": make_full_viewer_url(args, "semantic"),
            "object": make_full_viewer_url(args, "object"),
            "rgb": make_full_viewer_url(args, "rgb"),
        },
        "decision_template": str(args.output_dir / "manual_object_review_decisions.csv"),
        "decision_template_url": args.decision_template_url or "manual_object_review_decisions.csv",
        "objects": objects,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "semantic_object_review_index.json"
    html_path = args.output_dir / "semantic_object_review_index.html"
    decision_path = args.output_dir / "manual_object_review_decisions.csv"
    write_decision_template(objects, decision_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "html": str(html_path),
                "decision_template": str(decision_path),
                "object_count": len(objects),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
