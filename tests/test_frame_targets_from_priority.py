import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "build_frame_targets_from_priority_for_test",
        SCRIPTS / "build_frame_targets_from_priority.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_labels_skips_sky_and_residual_by_default():
    module = load_module()

    assert module.parse_labels(["ground", "wall", "sky", "0"], include_residual=False) == {1, 2}
    assert module.parse_labels(["ground", "0"], include_residual=True) == {0, 1}


def test_connected_components_returns_small_residuals():
    module = load_module()
    points = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.03, 0.00, 0.00],
            [1.00, 1.00, 1.00],
            [1.03, 1.00, 1.00],
            [4.00, 4.00, 4.00],
        ],
        dtype=np.float32,
    )

    comps, residual = module.connected_components(points, voxel_size=0.08, min_points=2)

    assert [len(comp) for comp in comps] == [2, 2]
    assert residual.tolist() == [False, False, False, False, True]


def test_target_summary_contains_fusion_required_fields():
    module = load_module()
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    colors = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50]], dtype=np.uint8)
    uu = np.array([10, 12, 14], dtype=np.int32)
    vv = np.array([20, 22, 24], dtype=np.int32)

    summary = module.target_summary(points, colors, uu, vv)

    assert summary["cluster_size"] == 3
    assert summary["bbox_3d"]["min"] == [0.0, 0.0, 0.0]
    assert summary["bbox_3d"]["max"] == [1.0, 1.0, 0.0]
    assert summary["bbox_2d"]["xyxy"] == [10, 20, 14, 24]
    assert summary["mean_color"] == [20.0, 30.0, 40.0]
    assert "normal" in summary["pca"]
