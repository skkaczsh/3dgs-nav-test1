#!/usr/bin/env python3
"""Build a static HTML index for cross-candidate merge review."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def score_text(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def write_decision_template(items: list[dict], path: Path) -> None:
    fields = [
        "review_id",
        "object_a",
        "object_b",
        "decision",
        "confidence",
        "reviewer",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in items:
            proposal = item["proposal"]
            writer.writerow(
                {
                    "review_id": item["review_id"],
                    "object_a": proposal["object_a"],
                    "object_b": proposal["object_b"],
                    "decision": "pending",
                    "confidence": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def render_item(item: dict, contact_sheet_dir: Path, output_dir: Path) -> str:
    proposal = item["proposal"]
    sheet = contact_sheet_dir / f"{item['review_id']}_contact_sheet.jpg"
    sheet_src = html.escape(rel(sheet, output_dir))
    rows = [
        ("objects", f"{proposal['object_a']} + {proposal['object_b']}"),
        ("candidates", f"{proposal['candidate_a']} / {proposal['candidate_b']}"),
        ("same source", str(proposal.get("same_source_cluster", ""))),
        ("score", score_text(proposal.get("score"))),
        ("centroid distance", score_text(proposal.get("centroid_distance"))),
        ("bbox distance", score_text(proposal.get("bbox_distance"))),
        ("bbox overlap", score_text(proposal.get("bbox_overlap_ratio"))),
        ("color distance", score_text(proposal.get("color_distance"))),
        ("points", f"{proposal.get('point_count_a', '')} / {proposal.get('point_count_b', '')}"),
        ("frames", f"{proposal.get('frame_min_a', '')}-{proposal.get('frame_max_a', '')} / {proposal.get('frame_min_b', '')}-{proposal.get('frame_max_b', '')}"),
    ]
    table = "\n".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
    )
    return f"""
<section class="item" id="{html.escape(item['review_id'])}">
  <header>
    <h2>{html.escape(item['review_id'])}</h2>
    <div class="decision">
      <span>decision:</span>
      <code>merge</code>
      <code>keep_split</code>
      <code>uncertain</code>
    </div>
  </header>
  <img src="{sheet_src}" alt="{html.escape(item['review_id'])} contact sheet" />
  <table>{table}</table>
</section>
"""


def render_html(items: list[dict], contact_sheet_dir: Path, output_dir: Path, decision_csv: Path) -> str:
    cards = "\n".join(render_item(item, contact_sheet_dir, output_dir) for item in items)
    decision_rel = html.escape(rel(decision_csv, output_dir))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cross-Candidate Merge Review</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f3; color: #202020; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .summary {{ margin-bottom: 24px; line-height: 1.45; max-width: 980px; }}
    .summary code, .decision code {{ background: #e8e8e4; border: 1px solid #d8d8d2; border-radius: 4px; padding: 2px 6px; }}
    .item {{ background: white; border: 1px solid #d8d8d2; border-radius: 8px; margin: 0 0 28px; padding: 16px; }}
    .item header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    .item h2 {{ margin: 0 0 12px; font-size: 20px; }}
    .decision {{ font-size: 14px; color: #444; }}
    img {{ display: block; max-width: 100%; height: auto; border: 1px solid #ddd; background: #fff; }}
    table {{ margin-top: 12px; border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #eee; text-align: left; padding: 6px 8px; }}
    th {{ width: 180px; color: #555; font-weight: 600; }}
  </style>
</head>
<body>
<main>
  <h1>Cross-Candidate Merge Review</h1>
  <div class="summary">
    <p><strong>Scene context:</strong> incremental MANIFOLD/Mid360 rooftop scan. A large roof/ground surface ratio is expected. Focus on railings, equipment boxes, pipes/cables, thin metal structures, and boundaries where fine objects touch large surfaces.</p>
    <p><strong>Decision rule:</strong> use <code>merge</code> only when both sides are the same physical object or continuous structure. Use <code>keep_split</code> for nearby/touching but distinct objects or coarse-mask overlap. Use <code>uncertain</code> if visual evidence is insufficient.</p>
    <p>Fill decisions in <a href="{decision_rel}">{decision_rel}</a>, then apply them with <code>apply_cross_candidate_merge_reviews.py</code>.</p>
  </div>
  {cards}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    items = load_jsonl(args.review_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision_csv = args.output_dir / "manual_merge_decisions.csv"
    write_decision_template(items, decision_csv)
    html_path = args.output_dir / "index.html"
    html_path.write_text(render_html(items, args.contact_sheet_dir, args.output_dir, decision_csv), encoding="utf-8")
    report = {
        "html": str(html_path),
        "decision_template": str(decision_csv),
        "item_count": len(items),
    }
    (args.output_dir / "review_html_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
