from pathlib import Path

from scripts.current_mainline_contract import REJECTED_ARTIFACT_SUBSTRINGS


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "tools" / "semantic_ply_viewer.html"


def test_semantic_ply_viewer_blocks_rejected_mainline_artifacts() -> None:
    html = VIEWER.read_text(encoding="utf-8")

    assert "REJECTED_ARTIFACT_SUBSTRINGS" in html
    assert "allowRejected" in html
    assert "Rejected diagnostic artifact blocked" in html
    for forbidden in REJECTED_ARTIFACT_SUBSTRINGS:
        assert forbidden in html
    assert "frame_object_points_stride10.ply" not in html


def test_semantic_ply_viewer_exposes_semantic_evidence_provenance() -> None:
    html = VIEWER.read_text(encoding="utf-8")

    assert "semanticEvidenceSourceScores" in html
    assert "semantic_evidence_source_scores" in html
    assert "semanticEvidenceScores" in html
    assert "semanticVetoedScores" in html
    assert "semanticFusionStatus" in html
    assert "Fusion 状态" in html
    assert "Evidence 来源" in html
    assert "Vetoed 分数" in html
