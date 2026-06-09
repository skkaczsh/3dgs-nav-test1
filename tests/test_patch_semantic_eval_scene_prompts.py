import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "patch_semantic_eval_scene_prompts_for_test",
        SCRIPTS / "patch_semantic_eval_scene_prompts.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_text_replaces_single_prompt_and_preserves_items_schema():
    module = load_module()
    source = '''PROMPT = """
old prompt
""".strip()

def keep():
    return 1
'''
    updated, count = module.patch_text(source, module.REVIEW_PROMPT)

    assert count == 1
    assert "rooftop MANIFOLD/Mid360 scan" in updated
    assert "mask-level evidence for dense point-cloud semantics" in updated
    assert "large stable surface layers" in updated
    assert "fine foreground targets" in updated
    assert '{"items":[{"mask_id":"1","label":"floor","confidence":0.90}]}' in updated
    assert "def keep()" in updated


def test_completion_prompt_preserves_surface_and_fine_target_goal():
    module = load_module()

    assert "fills non-sky semantic gaps for dense point-cloud projection" in module.COMPLETION_PROMPT
    assert "floor, wall, and building are large stable surface layers" in module.COMPLETION_PROMPT
    assert "railing, pipe, and equipment are fine foreground targets" in module.COMPLETION_PROMPT
    assert '{"items":[{"mask_id":"1","label":"floor","confidence":0.90}]}' in module.COMPLETION_PROMPT


def test_patch_file_creates_backup_once(tmp_path: Path):
    module = load_module()
    path = tmp_path / "review_merged_labels_prompt_v2.py"
    original = 'PROMPT = """\nold prompt\n""".strip()\n'
    path.write_text(original, encoding="utf-8")

    first = module.patch_file(path, module.REVIEW_PROMPT, dry_run=False)
    second = module.patch_file(path, module.REVIEW_PROMPT, dry_run=False)

    assert first["patched"] is True
    assert first["changed"] is True
    assert second["patched"] is True
    assert second["changed"] is False
    assert path.with_suffix(path.suffix + ".scene_prompt_bak").read_text(encoding="utf-8") == original
