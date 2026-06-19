from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_sync_time_models.py"
    spec = importlib.util.spec_from_file_location("analyze_sync_time_models", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nearest_candidate_metrics_reports_exact_rank_and_loss():
    module = load_module()
    rows = [
        {"video_idx": 12, "score": 0.9},
        {"video_idx": 10, "score": 0.5},
        {"video_idx": 11, "score": 0.8},
    ]
    metrics = module.nearest_candidate_metrics(rows, 11)
    assert metrics["present"] is True
    assert metrics["best_video_idx"] == 12
    assert metrics["exact_rank"] == 3
    assert abs(metrics["exact_score_loss"] - 0.1) < 1e-9
    assert metrics["nearest_distance"] == 0


def test_nearest_candidate_metrics_uses_nearest_when_exact_missing():
    module = load_module()
    rows = [
        {"video_idx": 20, "score": 0.7},
        {"video_idx": 10, "score": 0.9},
    ]
    metrics = module.nearest_candidate_metrics(rows, 18)
    assert metrics["present"] is False
    assert metrics["nearest_video_idx"] == 20
    assert metrics["nearest_distance"] == 2
    assert metrics["nearest_rank"] == 1
    assert metrics["nearest_score_loss"] == 0.0


def test_fit_line_handles_affine_mapping():
    module = load_module()
    slope, intercept = module.fit_line([0, 10, 20], [5, 25, 45])
    assert round(slope, 6) == 2.0
    assert round(intercept, 6) == 5.0
