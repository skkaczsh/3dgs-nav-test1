#!/usr/bin/env python3
"""Summarize the cross-candidate review stage into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def build_summary(pack_dir: Path, long_objects: Path | None = None) -> dict:
    review_pack = read_json(pack_dir / "cross_candidate_review_pack_report.json")
    contact_sheet = read_json(pack_dir / "contact_sheets" / "contact_sheet_report.json")
    html_report = read_json(pack_dir / "review_html" / "review_html_report.json")
    manual_norm = read_json(pack_dir / "manual_review_normalized" / "manual_merge_review_report.json")
    workflow = read_json(pack_dir / "manual_workflow_pending" / "manual_merge_workflow_report.json")
    merge_report = read_json(pack_dir / "manual_workflow_pending" / "applied" / "review_merge_report.json")
    merge_qa = read_json(pack_dir / "manual_workflow_pending" / "qa_reviewed_merge_report.json")
    vlm_report = read_json(pack_dir / "vlm_review_qwen_compact" / "vlm_merge_review_report.json")
    if "missing" in vlm_report:
        vlm_report = read_json(pack_dir / "vlm_review_qwen" / "vlm_merge_review_report.json")
    return {
        "pack_dir": str(pack_dir),
        "long_objects": str(long_objects) if long_objects else "",
        "review_items": count_jsonl(pack_dir / "cross_candidate_review_items.jsonl"),
        "contact_sheets": contact_sheet,
        "review_pack": review_pack,
        "html": html_report,
        "manual_normalized": manual_norm,
        "manual_workflow_pending": workflow,
        "manual_merge_report": merge_report,
        "manual_merge_qa": merge_qa,
        "qwen_vlm_review": vlm_report,
        "stage_status": {
            "review_pack_ready": review_pack.get("items_with_any_image") == review_pack.get("item_count"),
            "contact_sheets_ready": contact_sheet.get("sheet_count") == review_pack.get("item_count"),
            "manual_html_ready": "missing" not in html_report,
            "manual_decisions_pending": workflow.get("manual_review_count", 0) == 0,
            "pending_apply_safe": merge_report.get("accepted_merge_count", -1) == 0 and merge_qa.get("passed") is True,
            "qwen_review_ready": "missing" not in vlm_report and vlm_report.get("error_count", 0) == 0,
        },
    }


def render_markdown(summary: dict) -> str:
    status = summary["stage_status"]
    pack = summary["review_pack"]
    workflow = summary["manual_workflow_pending"]
    qa = summary["manual_merge_qa"]
    lines = [
        "# Cross-Candidate Review Stage Summary",
        "",
        "## Status",
        "",
        f"- review pack ready: `{status['review_pack_ready']}`",
        f"- contact sheets ready: `{status['contact_sheets_ready']}`",
        f"- manual HTML ready: `{status['manual_html_ready']}`",
        f"- manual decisions pending: `{status['manual_decisions_pending']}`",
        f"- pending apply safe: `{status['pending_apply_safe']}`",
        f"- Qwen VLM review ready: `{status['qwen_review_ready']}`",
        "",
        "## Counts",
        "",
        f"- review items: `{summary['review_items']}`",
        f"- items with image: `{pack.get('items_with_any_image', '')}`",
        f"- copied overlays: `{pack.get('copied_overlay_count', '')}`",
        f"- contact sheets: `{summary['contact_sheets'].get('sheet_count', '')}`",
        f"- manual reviews normalized: `{workflow.get('manual_review_count', '')}`",
        f"- manual errors/pending: `{workflow.get('manual_error_count', '')}`",
        f"- input objects: `{workflow.get('input_object_count', '')}`",
        f"- output objects: `{workflow.get('output_object_count', '')}`",
        f"- accepted merges: `{workflow.get('accepted_merge_count', '')}`",
        f"- QA passed: `{qa.get('passed', '')}`",
        "",
        "## Key Artifacts",
        "",
        f"- review HTML: `{summary['html'].get('html', '')}`",
        f"- manual CSV: `{summary['html'].get('decision_template', '')}`",
        f"- pending reviewed objects: `{summary['manual_merge_report'].get('objects_path', '')}`",
        f"- QA report: `{summary['pack_dir']}/manual_workflow_pending/qa_reviewed_merge_report.json`",
        "",
        "## Next Commands",
        "",
        "After filling `manual_merge_decisions.csv`:",
        "",
        "```bash",
        "python3 scripts/run_manual_merge_review_workflow.py \\",
        "  --manual-csv <review_html/manual_merge_decisions.csv> \\",
        "  --review-jsonl <cross_candidate_review_items.jsonl> \\",
        "  --objects <long_objects.jsonl> \\",
        "  --output-dir <manual_workflow_reviewed> \\",
        "  --min-confidence 0.75",
        "",
        "python3 scripts/qa_reviewed_merge_results.py \\",
        "  --input-objects <long_objects.jsonl> \\",
        "  --output-objects <manual_workflow_reviewed/applied/review_merged_long_objects.jsonl> \\",
        "  --decisions <manual_workflow_reviewed/applied/review_merge_decisions.jsonl> \\",
        "  --output-report <manual_workflow_reviewed/qa_reviewed_merge_report.json>",
        "```",
        "",
        "When server connectivity returns, rerun compact Qwen review before applying model decisions.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--long-objects", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args.pack_dir, args.long_objects)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "cross_candidate_review_stage_summary.json"
    md_path = args.output_dir / "cross_candidate_review_stage_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
