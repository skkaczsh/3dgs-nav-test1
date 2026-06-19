from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_video_frame_access.py"
    spec = importlib.util.spec_from_file_location("audit_video_frame_access", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_image_metrics_reports_identical_images():
    module = load_module()
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    metrics = module.image_metrics(image, image.copy())
    assert metrics["available"] is True
    assert metrics["mean_abs_diff"] == 0.0
    assert metrics["max_abs_diff"] == 0


def test_summarize_fails_when_threshold_exceeded():
    module = load_module()
    rows = [
        {"metric": {"available": True, "mean_abs_diff": 1.0, "gray_corr": 0.99}},
        {"metric": {"available": True, "mean_abs_diff": 3.0, "gray_corr": 0.95}},
    ]
    summary = module.summarize(rows, "metric", mad_threshold=2.0)
    assert summary["available_count"] == 2
    assert summary["mean_abs_diff"]["max"] == 3.0
    assert summary["pass"] is False
