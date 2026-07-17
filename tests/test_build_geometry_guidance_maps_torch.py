from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "build_geometry_guidance_maps_torch_for_test",
    SCRIPTS / "build_geometry_guidance_maps.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_torch_projector_requires_available_cuda(monkeypatch: pytest.MonkeyPatch):
    class NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class TorchWithoutCuda:
        cuda = NoCuda()

    monkeypatch.setattr(module.importlib, "import_module", lambda _name: TorchWithoutCuda())

    with pytest.raises(RuntimeError, match="available CUDA device"):
        module.require_torch_cuda()


def test_torch_global_projection_matches_numpy_depth_smoke():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Torch global projector smoke test")

    points_world = np.array(
        [
            [0.3, 0.1, 1.5],  # Near point at (4, 3).
            [0.8, 0.1, 3.5],  # Same pixel, farther: must lose z-buffering.
            [-0.6, 0.6, 1.5],
            [0.0, -0.4, 2.5],
            [0.0, 0.0, -2.0],  # Behind the camera.
            [20.0, 0.0, 1.5],  # Outside the image.
        ],
        dtype=np.float32,
    )
    world_to_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    camera_matrix = np.array(
        [[4.0, 0.0, 3.0], [0.0, 4.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    height = width = 8
    numpy_result = module.project_world_points_numpy(
        points_world, world_to_camera, camera_matrix, height, width, min_depth=0.1
    )
    torch_result = module.project_world_points_torch(
        points_world, world_to_camera, camera_matrix, height, width, min_depth=0.1
    )

    def depth_and_valid(result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        _idx, uu, vv, depths = result
        depth = np.zeros((height, width), dtype=np.float32)
        valid = np.zeros((height, width), dtype=bool)
        depth[vv, uu] = depths
        valid[vv, uu] = True
        return depth, valid

    numpy_depth, numpy_valid = depth_and_valid(numpy_result)
    torch_depth, torch_valid = depth_and_valid(torch_result)

    assert np.array_equal(torch_valid, numpy_valid)
    assert np.allclose(torch_depth[numpy_valid], numpy_depth[numpy_valid], rtol=1e-5, atol=1e-5)
    assert numpy_depth[3, 4] == pytest.approx(2.0)
