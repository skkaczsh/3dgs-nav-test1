import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_prompt_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("vlm_scene_prompt_for_test", SCRIPTS / "vlm_scene_prompt.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mask_label_prompt_contains_scene_taxonomy_and_strict_json():
    module = load_prompt_module()
    prompt = module.mask_label_prompt()

    assert "rooftop" in prompt.lower()
    assert "Return only strict JSON" in prompt
    assert "railing" in prompt
    assert "equipment" in prompt
    assert "mixed" in prompt
    assert "Do not invent labels or synonyms" in prompt
    assert "dense point-level semantics" in prompt
    assert "large stable surface layers" in prompt
    assert "fine foreground targets" in prompt
    assert "description" in prompt
    assert "identity_hint" in prompt
    assert "white HVAC outdoor unit" in prompt
    assert "Never copy overlay colors" in prompt


def test_merge_review_prompt_includes_candidate_metadata_and_failure_mode():
    module = load_prompt_module()
    prompt = module.merge_review_prompt(
        {
            "review_id": "review_001",
            "proposal": {
                "object_a": "obj_a",
                "object_b": "obj_b",
                "candidate_a": "cand_a",
                "candidate_b": "cand_b",
                "score": 0.9,
                "centroid_distance": 0.2,
                "bbox_distance": 0.1,
                "bbox_overlap_ratio": 0.3,
                "color_distance": 35.0,
                "same_source_cluster": False,
            },
        }
    )

    assert "review_001" in prompt
    assert "obj_a" in prompt
    assert "light brown/gray roof pixels" in prompt
    assert "merge | keep_split | uncertain" in prompt
    assert "Build dense point-level semantics" in prompt
