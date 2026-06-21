import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_v2_v8_hybrid_preview import hybrid_label  # noqa: E402


def test_v2_surface_vetoes_unconfirmed_v8_car():
    label, reason = hybrid_label("car", "wall")
    assert label == "wall"
    assert reason == "v2_surface_overrides_v8_car"


def test_v2_vegetation_overrides_v8_surface_or_unknown():
    assert hybrid_label("wall", "grass")[0] == "grass"
    assert hybrid_label("unknown", "tree")[0] == "tree"


def test_v8_ceiling_is_preserved_over_v2_teacher():
    label, reason = hybrid_label("ceiling", "floor")
    assert label == "ceiling"
    assert reason == "keep_v8_ceiling"


def test_unconfirmed_fine_label_becomes_candidate_without_v2_support():
    label, reason = hybrid_label("railing", "unknown")
    assert label == "fine_candidate"
    assert reason == "v8_railing_unconfirmed"
