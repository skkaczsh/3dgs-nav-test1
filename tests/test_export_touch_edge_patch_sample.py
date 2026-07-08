import json

import numpy as np

from scripts.export_touch_edge_patch_sample import main
from scripts.optimize_patch_graph_energy import read_labels, write_labels


def write_region_input(path, n, src, dst):
    arrays = [
        np.arange(n * 3, dtype="<f4").reshape(n, 3),
        np.zeros((n, 3), dtype="<f4"),
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
        np.array([len(src)], dtype="<i8").tofile(f)
        for value in arrays:
            value.tofile(f)
        np.asarray(src, dtype="<i4").tofile(f)
        np.asarray(dst, dtype="<i4").tofile(f)


def test_export_touch_edge_patch_sample_writes_aligned_labels(tmp_path, monkeypatch):
    region = tmp_path / "region.bin"
    labels = tmp_path / "labels.bin"
    out = tmp_path / "out"
    write_region_input(region, 5, [0, 2], [2, 4])
    write_labels(labels, np.array([1, 1, 2, 2, 3], dtype=np.int32))

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_touch_edge_patch_sample.py",
            "--region-input",
            str(region),
            "--labels",
            str(labels),
            "--output-dir",
            str(out),
            "--max-per-patch",
            "1",
        ],
    )
    assert main() == 0

    report = json.loads((out / "touch_edge_patch_sample_report.json").read_text())
    assert report["touch_edge_count"] == 2
    assert report["target_patch_count"] == 3
    assert report["sample_point_count"] == 3
    assert sorted(read_labels(out / "touch_edge_patch_sample_labels.bin").tolist()) == [1, 2, 3]
