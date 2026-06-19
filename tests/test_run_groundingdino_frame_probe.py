from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "run_groundingdino_frame_probe_for_test",
    SCRIPTS / "run_groundingdino_frame_probe.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_parse_prompt_args_overrides_label_group():
    prompts = module.parse_prompt_args(["railing=guardrail, handrail"])

    assert prompts["railing"] == ["guardrail", "handrail"]
    assert "car" in prompts


def test_canonical_label_matches_prompt_text():
    prompts = {"railing": ["metal fence", "handrail"], "car": ["parked car"]}

    assert module.canonical_label("a metal fence", prompts) == "railing"
    assert module.canonical_label("parked car", prompts) == "car"
    assert module.canonical_label("tree", prompts) == "unknown"


def test_detection_stats_counts_area_ratios():
    stats = module.detection_stats(
        [
            {"label": "railing", "bbox_xyxy": [0, 0, 10, 10]},
            {"label": "railing", "bbox_xyxy": [10, 10, 20, 20]},
            {"label": "car", "bbox_xyxy": [0, 0, 5, 5]},
        ],
        (100, 100),
        large_box_ratio=0.01,
    )

    assert stats["detection_counts"] == {"railing": 2, "car": 1}
    assert stats["box_area_pixels"] == {"railing": 200, "car": 25}
    assert stats["box_area_ratios"]["railing"] == 0.02
    assert stats["large_box_counts"] == {"railing": 2}


def test_detection_stats_clips_boxes_to_image_bounds():
    stats = module.detection_stats(
        [{"label": "railing", "bbox_xyxy": [-10, -10, 150, 150]}],
        (100, 100),
        large_box_ratio=0.5,
    )

    assert stats["box_area_pixels"]["railing"] == 10000
    assert stats["box_area_ratios"]["railing"] == 1.0


def test_aggregate_detection_summary_reports_large_boxes():
    summary = module.aggregate_detection_summary(
        [
            {
                "detection_counts": {"railing": 2},
                "large_box_counts": {"railing": 1},
                "box_area_ratios": {"railing": 0.4},
            },
            {
                "detection_counts": {"railing": 1, "car": 1},
                "large_box_counts": {"car": 1},
                "box_area_ratios": {"railing": 0.2, "car": 0.5},
            },
        ]
    )

    assert summary["detection_counts"] == {"railing": 3, "car": 1}
    assert summary["large_box_counts"] == {"railing": 1, "car": 1}
    assert abs(summary["mean_box_area_ratio_by_label"]["railing"] - 0.2) < 1e-9
    assert summary["max_box_area_ratio_by_label"]["car"] == 0.5
