from pathlib import Path


def test_start_viewer_refreshes_index_and_prints_index_url() -> None:
    script = Path("scripts/start_semantic_ply_viewer.sh").read_text(encoding="utf-8")
    assert "build_semantic_viewer_index.py" in script
    assert "index_url=" in script
    assert "semantic_viewer_index.html" in script
