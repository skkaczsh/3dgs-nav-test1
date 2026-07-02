import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_spatial_partition_objects import connected_components_by_label, main, select_voxel_labels  # noqa: E402


def test_each_voxel_gets_one_winning_label():
    voxel_points = {
        (0, 0, 0): {"count": 1},
        (1, 0, 0): {"count": 1},
    }
    votes = {
        (0, 0, 0): Counter({"wall": 2, "floor": 1}),
        (1, 0, 0): Counter({"floor": 3}),
    }

    labels = select_voxel_labels(voxel_points, votes, "unknown")

    assert labels == {(0, 0, 0): "wall", (1, 0, 0): "floor"}


def test_disconnected_same_label_voxels_become_separate_objects():
    voxel_labels = {
        (0, 0, 0): "wall",
        (1, 0, 0): "wall",
        (10, 0, 0): "wall",
    }

    object_for_voxel, objects, _report = connected_components_by_label(voxel_labels, {"*": 1}, "keep")

    assert len(objects) == 2
    assert object_for_voxel[(0, 0, 0)] == object_for_voxel[(1, 0, 0)]
    assert object_for_voxel[(10, 0, 0)] != object_for_voxel[(0, 0, 0)]


def test_small_components_are_kept_by_default_and_marked():
    voxel_labels = {
        (0, 0, 0): "railing",
        (5, 0, 0): "railing",
        (6, 0, 0): "railing",
    }

    object_for_voxel, objects, report = connected_components_by_label(voxel_labels, {"*": 1, "railing": 2}, "keep")

    assert len(objects) == 2
    assert (0, 0, 0) in object_for_voxel
    assert report["dropped_small_voxels_by_label"] == {}
    assert [obj["status"] for obj in sorted(objects, key=lambda item: item["voxel_count"])] == [
        "small_component",
        "spatial_connected_component",
    ]


def test_small_components_can_be_dropped_for_filtered_previews():
    voxel_labels = {
        (0, 0, 0): "railing",
        (5, 0, 0): "railing",
        (6, 0, 0): "railing",
    }

    object_for_voxel, objects, report = connected_components_by_label(voxel_labels, {"*": 1, "railing": 2}, "drop")

    assert len(objects) == 1
    assert (0, 0, 0) not in object_for_voxel
    assert report["dropped_small_voxels_by_label"] == {"railing": 1}


def test_spatial_partition_rejects_stride_preview_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_spatial_partition_objects.py",
            "--base-ply",
            str(tmp_path / "dense_source.ply"),
            "--teacher",
            f"qa:{tmp_path / 'frame_object_points_stride10.ply'}:1.0",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(ValueError, match="forbidden input path"):
        main()
