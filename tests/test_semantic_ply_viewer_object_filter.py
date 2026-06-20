from pathlib import Path


def test_semantic_ply_viewer_supports_object_url_filter() -> None:
    html = Path("tools/semantic_ply_viewer.html").read_text(encoding="utf-8")

    assert 'id="objectFilter"' in html
    assert 'params.get("object")' in html
    assert "applyUrlObjectParam(params)" in html
    assert "objectKeys(row.objectId)" in html
    assert "Object 筛选" in html
