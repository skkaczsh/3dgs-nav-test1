from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.update_spg_visual_acceptance import format_md, recompute_status, update_record


def test_update_spg_visual_acceptance_recomputes_status(tmp_path: Path) -> None:
    acceptance = tmp_path / "spg.json"
    acceptance.write_text(
        json.dumps(
            {
                "schema": "superpoint-graph-visual-acceptance/v1",
                "status": "pending",
                "candidate": "spg",
                "checks": [
                    {"id": "a", "required": True, "status": "pending", "question": "A?"},
                    {"id": "b", "required": True, "status": "accepted", "question": "B?"},
                ],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        acceptance=acceptance,
        check_id="a",
        status="accepted",
        notes="ok",
        reviewer="tester",
    )

    record = update_record(args)

    assert record["status"] == "accepted"
    assert record["checks"][0]["notes"] == "ok"
    assert record["reviewer"] == "tester"
    assert "Run `python3 scripts/validate_current_mainline.py`" in format_md(record)


def test_format_md_uses_custom_title_and_viewer() -> None:
    record = {
        "title": "Custom SPG Acceptance",
        "status": "pending",
        "candidate": "spg_candidate",
        "review_doc": "docs/review.md",
        "viewer_url": "http://127.0.0.1:8765/viewer",
        "checks": [
            {"id": "a", "required": True, "status": "pending", "question": "A?"},
        ],
    }

    text = format_md(record)

    assert text.startswith("# Custom SPG Acceptance")
    assert "Viewer: http://127.0.0.1:8765/viewer" in text


def test_spg_visual_acceptance_failed_required_blocks_record() -> None:
    record = {
        "checks": [
            {"id": "a", "required": True, "status": "accepted"},
            {"id": "b", "required": True, "status": "failed"},
        ]
    }

    assert recompute_status(record) == "failed"
