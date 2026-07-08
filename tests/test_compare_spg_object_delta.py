from __future__ import annotations

import json
import struct
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_spg_object_delta import compare, markdown


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_labels(path: Path, labels: list[int]) -> None:
    with path.open("wb") as f:
        f.write(b"GPRGlabels1\n")
        f.write(struct.pack("<q", len(labels)))
        array("i", labels).tofile(f)


def test_compare_spg_object_delta_finds_new_candidate_merges(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(
        baseline,
        [
            {"patch_id": 1, "object": 1, "voxel_count": 10, "source_patch_ids": [1]},
            {"patch_id": 2, "object": 2, "voxel_count": 20, "source_patch_ids": [2]},
        ],
    )
    write_jsonl(
        candidate,
        [
            {
                "patch_id": 1,
                "object": 1,
                "voxel_count": 30,
                "geometry_type": "vertical",
                "bucket_entropy": 0.2,
                "source_patch_count": 2,
                "source_patch_ids": [1, 2],
            }
        ],
    )

    report = compare(baseline, candidate)

    assert report["new_merge_object_count"] == 1
    assert report["top_new_merges"][0]["baseline_objects_merged"] == [1, 2]
    assert "candidate object" in markdown(report)


def test_compare_spg_object_delta_prefers_label_overlap(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    baseline_labels = tmp_path / "baseline.bin"
    candidate_labels = tmp_path / "candidate.bin"
    write_jsonl(
        baseline,
        [
            {"patch_id": 1, "object": 1, "voxel_count": 2, "source_patch_ids": [1]},
            {"patch_id": 2, "object": 2, "voxel_count": 2, "source_patch_ids": [2]},
        ],
    )
    write_jsonl(
        candidate,
        [
            {
                "patch_id": 1,
                "object": 1,
                "voxel_count": 4,
                "geometry_type": "horizontal",
                "bucket_entropy": 0,
                "source_patch_ids": [1],
            }
        ],
    )
    write_labels(baseline_labels, [1, 1, 2, 2])
    write_labels(candidate_labels, [1, 1, 1, 1])

    report = compare(baseline, candidate, baseline_labels=baseline_labels, candidate_labels=candidate_labels)

    assert report["method"] == "labels"
    assert report["new_merge_object_count"] == 1
    assert report["top_new_merges"][0]["baseline_objects_merged"] == [1, 2]
    assert report["top_new_merges"][0]["baseline_object_voxel_counts"] == [
        {"object": 1, "voxels": 2},
        {"object": 2, "voxels": 2},
    ]
