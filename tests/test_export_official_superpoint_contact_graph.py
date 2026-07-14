import numpy as np

from scripts.export_official_superpoint_contact_graph import contact_rows


def test_contact_graph_contains_only_true_neighbor_faces() -> None:
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)
    labels = np.array([0, 1, 1, 2], dtype=np.int32)
    rows, stats = contact_rows(xyz, labels, 1.0, 1)
    assert rows == [{"object_a": 0, "object_b": 1, "shared_voxel_faces": 1, "contact_ratio_min": 1.0}]
    assert stats["kept_contact_pairs"] == 1
