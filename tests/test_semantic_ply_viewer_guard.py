from pathlib import Path

from scripts.current_mainline_contract import FORBIDDEN_ARTIFACT_SUBSTRINGS


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "tools" / "semantic_ply_viewer.html"


def test_semantic_ply_viewer_blocks_rejected_mainline_artifacts() -> None:
    html = VIEWER.read_text(encoding="utf-8")

    assert "REJECTED_ARTIFACT_SUBSTRINGS" in html
    assert "allowRejected" in html
    assert "Rejected diagnostic artifact blocked" in html
    for forbidden in FORBIDDEN_ARTIFACT_SUBSTRINGS:
        assert forbidden in html

