from __future__ import annotations

import json
from pathlib import Path

from scripts import compare_geo_patch_runs as module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_summarize_jsonl_flags_large_mixed_patches(tmp_path: Path) -> None:
    jsonl = tmp_path / "patches.jsonl"
    write_jsonl(
        jsonl,
        [
            {
                "patch_id": 1,
                "voxel_count": 20000,
                "geometry_type": "mixed",
                "bucket_counts": {"horizontal": 8000, "vertical": 8000, "unknown": 4000},
                "bucket_entropy": 1.5,
                "extent": [100.0, 1.0, 0.5],
                "conflict_flags": ["high_entropy"],
            },
            {
                "patch_id": 2,
                "voxel_count": 100,
                "geometry_type": "rough_mixed",
                "bucket_counts": {"rough_mixed": 100},
                "bucket_entropy": 0.0,
                "extent": [1.0, 1.0, 1.0],
            },
        ],
    )

    report = module.summarize_jsonl(jsonl, large_voxels=10000, entropy_threshold=1.1)

    assert report["patch_count"] == 2
    assert report["high_entropy_count"] == 1
    assert report["large_high_entropy_count"] == 1
    assert report["large_low_purity_count"] == 1
    assert report["large_extreme_aspect_count"] == 1
    assert report["top_patches"][0]["patch_id"] == 1


def test_summarize_jsonl_computes_missing_bucket_entropy(tmp_path: Path) -> None:
    jsonl = tmp_path / "patches.jsonl"
    write_jsonl(
        jsonl,
        [
            {
                "patch_id": 1,
                "voxel_count": 10,
                "geometry_type": "mixed",
                "bucket_counts": {"horizontal": 5, "vertical": 5},
                "extent": [1.0, 1.0, 1.0],
            }
        ],
    )

    report = module.summarize_jsonl(jsonl, large_voxels=100, entropy_threshold=0.9)

    assert report["high_entropy_count"] == 1
    assert report["top_patches"][0]["bucket_entropy"] == 1.0


def test_summarize_merge_log_counts_profiles(tmp_path: Path) -> None:
    merge_log = tmp_path / "merge.jsonl"
    write_jsonl(
        merge_log,
        [
            {"status": "accept", "reason": "accepted_fragment_attachment"},
            {"status": "accept", "reason": "accepted_attachment"},
            {"status": "reject", "reason": "fragment_attachment_score"},
        ],
    )

    summary = module.summarize_merge_log(merge_log)

    assert summary["status_counts"] == {"accept": 2, "reject": 1}
    assert summary["accepted_profiles"]["accepted_fragment_attachment"] == 1
    assert summary["reason_counts"]["fragment_attachment_score"] == 1


def test_build_markdown_uses_report_accepted_edges_without_merge_log() -> None:
    markdown = module.build_markdown(
        {
            "runs": [
                {
                    "name": "spg",
                    "patch_count": 2,
                    "high_entropy_count": 0,
                    "large_high_entropy_count": 0,
                    "large_low_purity_count": 0,
                    "source_report": {"accepted_edges": 7, "accepted_reasons": {"near_bbox_bridge": 3}},
                    "merge_log_summary": {},
                    "voxel_count_p50": 1,
                    "voxel_count_p90": 1,
                    "voxel_count_p99": 1,
                    "voxel_count_max": 1,
                    "bucket_entropy_p50": 0,
                    "bucket_entropy_p90": 0,
                    "bucket_entropy_p99": 0,
                    "top_patches": [],
                }
            ]
        }
    )

    assert "| spg | 2 | 0 | 0 | 0 | 7 | near_bbox_bridge:3 |" in markdown
