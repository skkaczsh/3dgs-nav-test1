from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_parking_frame_local_best_route.sh"


def test_frame_local_best_route_marks_local_geometry_split_as_qa_preview_input() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    viewer_pos = text.index("prepare_viewer_objects_report.json")
    candidate_pos = text.index("scripts/build_local_geometry_split_candidates.py")
    split_pos = text.index("scripts/split_priority_objects_by_local_geometry.py")
    qa_flag_pos = text.index("--allow-qa-preview-source", split_pos)

    assert viewer_pos < candidate_pos < split_pos < qa_flag_pos
    assert "--input-ply \"${qa_viewer_dir}/frame_object_points_stride10.ply\"" in text
