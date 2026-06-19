from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "build_fine_mask_eval_manifest_for_test",
    SCRIPTS / "build_fine_mask_eval_manifest.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def row(object_id, label, risk, cluster, frame=100, cam=1):
    return {
        "object_id": object_id,
        "semantic_label": label,
        "risk_score": risk,
        "risk_reasons": ["railing_not_linear"],
        "target_id": f"pt_{frame:06d}_cam{cam}_p5",
        "target_label": label,
        "frame_id": frame,
        "cam_id": cam,
        "cluster_size": cluster,
        "bbox_2d": {"xyxy": [0, 0, 10, 20]},
        "image_path": f"/frames/cam{cam}/frame_{frame:06d}.jpg",
        "mask_path": f"/masks/cam{cam}_{frame:06d}.png",
    }


def test_select_rows_filters_labels_and_limits_per_object():
    rows = [
        row(1, "railing", 80, 100, frame=100),
        row(1, "railing", 70, 200, frame=110),
        row(1, "railing", 60, 300, frame=120),
        row(2, "car", 90, 50, frame=130),
        row(3, "wall", 100, 1000, frame=140),
    ]

    selected = module.select_rows(rows, ["railing", "car"], limit=10, per_object_limit=2)

    assert [item["object_id"] for item in selected] == [2, 1, 1]
    assert all(item["semantic_label"] != "wall" for item in selected)


def test_build_manifest_writes_stable_sample_ids(tmp_path: Path):
    evidence = tmp_path / "evidence.jsonl"
    rows = [row(7, "railing", 100, 13071, frame=3480, cam=1)]
    evidence.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    args = SimpleNamespace(evidence_jsonl=evidence, labels=["railing"], limit=10, per_object_limit=3)

    manifest = module.build_manifest(module.read_jsonl(evidence), args)

    assert manifest["sample_count"] == 1
    assert manifest["object_count"] == 1
    assert manifest["items"][0]["sample_id"] == "0001_obj7_cam1_frame003480"
    assert manifest["items"][0]["recommended_eval"][0] == "use undistorted image as input"


def test_markdown_contains_gate_section(tmp_path: Path):
    args = SimpleNamespace(evidence_jsonl=tmp_path / "evidence.jsonl", labels=["railing"], limit=10, per_object_limit=3)
    manifest = module.build_manifest([row(7, "railing", 100, 13071)], args)

    text = module.markdown(manifest)

    assert "Fine Mask Evaluation Manifest" in text
    assert "Next Evaluation Gate" in text
