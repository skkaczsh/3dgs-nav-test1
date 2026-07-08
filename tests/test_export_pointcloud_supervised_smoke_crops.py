import numpy as np

from scripts.export_pointcloud_supervised_smoke_crops import export_crops
from scripts.optimize_patch_graph_energy import read_labels, write_labels


def write_region_input(path, xyz, rgb):
    n = len(xyz)
    arrays = [
        np.asarray(xyz, dtype="<f4"),
        np.asarray(rgb, dtype="<f4"),
        np.zeros((n, 3), dtype="<f4"),
        np.zeros(n, dtype="<f4"),
        np.zeros(n, dtype="<f4"),
        np.zeros(n, dtype="<f4"),
        np.zeros(n, dtype="<f4"),
        np.zeros(n, dtype="<f4"),
        np.zeros(n, dtype="<i2"),
    ]
    with path.open("wb") as f:
        f.write(b"GPRGv1\n")
        np.array([n], dtype="<i8").tofile(f)
        np.array([0], dtype="<i8").tofile(f)
        for value in arrays:
            value.tofile(f)


def test_export_crops_can_write_region_aligned_labels(tmp_path):
    region = tmp_path / "region.bin"
    labels = tmp_path / "labels.bin"
    output = tmp_path / "crops"
    dense = tmp_path / "unused.ply"
    dense.write_text("ply\n", encoding="utf-8")
    write_region_input(region, [[0, 0, 0], [2, 0, 0]], [[10, 20, 30], [40, 50, 60]])
    write_labels(labels, np.array([7, 8], dtype=np.int32))
    manifest = {
        "schema": "pointcloud-supervised-baseline-smoke-manifest/v1",
        "dense_input": {"ply": str(dense)},
        "crops": [
            {
                "id": "a",
                "geometry_type": "horizontal",
                "dense_voxel_count_in_crop": 1,
                "bbox_3d": {"min": [-1, -1, -1], "max": [1, 1, 1]},
            }
        ],
    }

    report = export_crops(manifest, output, labels, region)
    label_path = output / "a_labels.bin"
    assert report["crops"][0]["point_count"] == 1
    assert report["crops"][0]["output_labels"] == str(label_path)
    assert read_labels(label_path).tolist() == [7]
