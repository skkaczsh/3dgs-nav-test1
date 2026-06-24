from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.prepare_current_dense_visual_acceptance import build_record
from scripts.update_current_dense_visual_acceptance import recompute_status, update_record


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "update_current_dense_visual_acceptance.py"


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def qa(path: Path) -> Path:
    return write_json(
        path,
        {
            "object_refinement": {
                "metrics": {
                    "delta_v8_minus_v7": {
                        "accepted_candidate_rows": 1,
                        "output_object_count": -1,
                        "mixed_object_voxel_ratio_020": -0.1,
                    }
                }
            },
            "surface_guard": {"label_point_counts": {"delta_v17_minus_v9": {"floor": 0}}},
        },
    )


def make_acceptance(tmp_path: Path) -> Path:
    qa_path = qa(tmp_path / "qa.json")
    args = argparse.Namespace(
        qa_json=qa_path,
        output=tmp_path / "acceptance.json",
        output_md=tmp_path / "acceptance.md",
        review_index_url="http://127.0.0.1:8765/docs/current_dense_review_index.html",
        force=True,
    )
    return write_json(args.output, build_record(args))


def update_args(path: Path, check_id: str, status: str, **overrides):
    base = dict(
        acceptance=path,
        markdown=path.with_suffix(".md"),
        gate_output=path.with_name("gate.json"),
        check_id=check_id,
        status=status,
        notes=None,
        evidence=[],
        reviewer=None,
        summary=None,
        run_gate=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_update_one_required_check_keeps_record_pending(tmp_path: Path) -> None:
    path = make_acceptance(tmp_path)

    record = update_record(update_args(path, "v8_fragmentation_improves", "accepted", reviewer="tester"))

    assert record["status"] == "pending"
    assert record["reviewer"] == "tester"
    assert any(row["id"] == "v8_fragmentation_improves" and row["status"] == "accepted" for row in record["checks"])


def test_recompute_status_accepts_only_when_all_required_checks_accepted(tmp_path: Path) -> None:
    path = make_acceptance(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))

    for row in record["checks"]:
        row["status"] = "accepted"

    assert recompute_status(record) == "accepted"


def test_update_unknown_check_id_fails(tmp_path: Path) -> None:
    path = make_acceptance(tmp_path)

    with pytest.raises(ValueError, match="check id"):
        update_record(update_args(path, "missing", "accepted"))


def test_cli_updates_acceptance_and_markdown(tmp_path: Path) -> None:
    path = make_acceptance(tmp_path)
    md = tmp_path / "acceptance.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--acceptance",
            str(path),
            "--markdown",
            str(md),
            "--check-id",
            "v8_fragmentation_improves",
            "--status",
            "accepted",
            "--notes",
            "looks cleaner",
            "--reviewer",
            "tester",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "pending"
    assert "looks cleaner" in md.read_text(encoding="utf-8")
