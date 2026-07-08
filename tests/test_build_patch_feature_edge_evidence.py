import csv
import json

import numpy as np

from scripts.build_patch_feature_edge_evidence import main
from scripts.optimize_patch_graph_energy import write_labels


def write_region_input(path, n, src, dst):
    arrays = {
        "xyz": np.zeros((n, 3), dtype="<f4"),
        "rgb": np.zeros((n, 3), dtype="<f4"),
        "normal": np.zeros((n, 3), dtype="<f4"),
        "roughness": np.zeros(n, dtype="<f4"),
        "planarity": np.zeros(n, dtype="<f4"),
        "linearity": np.zeros(n, dtype="<f4"),
        "local_color_std": np.zeros(n, dtype="<f4"),
        "height_range": np.zeros(n, dtype="<f4"),
        "buckets": np.zeros(n, dtype="<i2"),
    }
    with path.open("wb") as f:
        f.write(b"GPRGv1\n")
        np.array([n], dtype="<i8").tofile(f)
        np.array([len(src)], dtype="<i8").tofile(f)
        for value in arrays.values():
            value.tofile(f)
        np.asarray(src, dtype="<i4").tofile(f)
        np.asarray(dst, dtype="<i4").tofile(f)


def test_build_patch_feature_edge_evidence_writes_touch_edge_csv(tmp_path, monkeypatch):
    region = tmp_path / "region.bin"
    labels = tmp_path / "labels.bin"
    features = tmp_path / "features.npz"
    out = tmp_path / "edges.csv"
    write_region_input(region, 3, [0, 1], [1, 2])
    write_labels(labels, np.array([1, 2, 3], dtype=np.int32))
    np.savez_compressed(
        features,
        patch_ids=np.array([1, 2], dtype=np.int64),
        features=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_patch_feature_edge_evidence.py",
            "--region-input",
            str(region),
            "--labels",
            str(labels),
            "--patch-features",
            str(features),
            "--output",
            str(out),
        ],
    )
    assert main() == 0

    rows = list(csv.DictReader(out.open()))
    report = json.loads((tmp_path / "edges.csv.report.json").read_text())
    assert rows == [{"patch_a": "1", "patch_b": "2", "similarity": "1.0", "contact_points": "1"}]
    assert report["touch_edge_count"] == 2
    assert report["written_edge_count"] == 1
    assert report["missing_feature_edge_count"] == 1
