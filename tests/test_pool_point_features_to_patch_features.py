import json

import numpy as np

from scripts.optimize_patch_graph_energy import write_labels
from scripts.pool_point_features_to_patch_features import main


def test_pool_point_features_to_patch_features(tmp_path, monkeypatch):
    labels = tmp_path / "labels.bin"
    point_features = tmp_path / "point_features.npz"
    output = tmp_path / "patch_features.npz"
    write_labels(labels, np.array([2, 1, 2], dtype=np.int32))
    np.savez_compressed(point_features, features=np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=np.float32))

    monkeypatch.setattr(
        "sys.argv",
        ["pool_point_features_to_patch_features.py", "--labels", str(labels), "--point-features", str(point_features), "--output", str(output)],
    )
    assert main() == 0

    out = np.load(output)
    report = json.loads((tmp_path / "patch_features.npz.report.json").read_text())
    assert out["patch_ids"].tolist() == [1, 2]
    assert out["counts"].tolist() == [1, 2]
    np.testing.assert_allclose(out["features"], [[0.0, 1.0], [1.0, 0.0]])
    assert report["patch_count"] == 2


def test_pool_point_features_can_use_point_indices(tmp_path, monkeypatch):
    labels = tmp_path / "labels.bin"
    point_features = tmp_path / "point_features.npz"
    output = tmp_path / "patch_features.npz"
    write_labels(labels, np.array([9, 1, 9, 2], dtype=np.int32))
    np.savez_compressed(
        point_features,
        point_indices=np.array([1, 3], dtype=np.int64),
        features=np.array([[0.0, 3.0], [4.0, 0.0]], dtype=np.float32),
    )

    monkeypatch.setattr(
        "sys.argv",
        ["pool_point_features_to_patch_features.py", "--labels", str(labels), "--point-features", str(point_features), "--output", str(output)],
    )
    assert main() == 0

    out = np.load(output)
    report = json.loads((tmp_path / "patch_features.npz.report.json").read_text())
    assert out["patch_ids"].tolist() == [1, 2]
    np.testing.assert_allclose(out["features"], [[0.0, 1.0], [1.0, 0.0]])
    assert report["used_point_indices"] is True
